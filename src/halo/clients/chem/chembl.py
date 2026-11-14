import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from rdkit.Chem import inchi

from halo.chem.mol import canonicalize_smiles, normalize_inchikey, to_mol
from halo.clients.http import (
    HttpClient,
    ResponseError,
    ServiceClient,
    ServiceError,
    check_integer,
    flag,
    integer,
    json_object,
    number,
    records,
    text,
)
from halo.config.settings import DEFAULT_INTEGRATIONS
from halo.core.errors import ParseError
from halo.models.analysis import ResultRelation
from halo.models.research import publication_number

logger = logging.getLogger(__name__)
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
STANDARD_RELATIONS = get_args(ResultRelation)
MAX_PAGE_SIZE = 1000

ClinicalPhase = Literal[
    "approved",
    "phase_3",
    "phase_2",
    "phase_1",
    "early_phase_1",
    "unknown_phase",
    "preclinical",
]
_PHASES: dict[float | None, ClinicalPhase] = {
    4.0: "approved",
    3.0: "phase_3",
    2.0: "phase_2",
    1.0: "phase_1",
    0.5: "early_phase_1",
    -1.0: "unknown_phase",
    None: "preclinical",
}
_MOLECULE_FIELDS = (
    "molecule_chembl_id,pref_name,molecule_structures,molecule_hierarchy,"
    "max_phase,therapeutic_flag,oral,withdrawn_flag"
)
_ACTIVITY_FIELDS = (
    "activity_id,molecule_chembl_id,parent_molecule_chembl_id,target_chembl_id,"
    "target_pref_name,target_organism,standard_type,standard_relation,standard_value,"
    "standard_units,pchembl_value,assay_chembl_id,assay_type,assay_description,"
    "data_validity_comment,potential_duplicate,document_chembl_id,document_journal,"
    "document_year,ligand_efficiency"
)
_ID_PATTERN = re.compile(r"CHEMBL[1-9]\d*")


@dataclass(frozen=True, slots=True)
class ChEMBLMolecule:
    """A ChEMBL molecule with locally canonicalized SMILES.

    Salts and mixtures have their own records linked to a parent.
    """

    chembl_id: str
    name: str | None
    smiles: str | None
    inchikey: str | None
    parent_chembl_id: str | None
    phase: ClinicalPhase
    therapeutic: bool | None
    oral: bool | None
    withdrawn: bool | None


@dataclass(frozen=True, slots=True)
class Bioactivity:
    """A reported measurement with its assay and source.

    standard_relation distinguishes exact values from bounds. pchembl_value
    is ChEMBL's negative log10 of an exact potency in molar units, when provided.
    """

    activity_id: int
    molecule_chembl_id: str
    parent_molecule_chembl_id: str | None
    target_chembl_id: str | None
    target_name: str | None
    organism: str | None
    standard_type: str | None
    standard_relation: ResultRelation | None
    standard_value: float | None
    standard_units: str | None
    pchembl_value: float | None
    assay_chembl_id: str | None
    assay_type: str | None
    assay_description: str | None
    validity_comment: str | None
    potential_duplicate: bool | None
    document_chembl_id: str | None
    journal: str | None
    year: int | None
    ligand_efficiency: float | None
    lipophilic_efficiency: float | None
    binding_efficiency_index: float | None


@dataclass(frozen=True)
class CompoundActivities:
    """Retrieved measurements, the server count, and repeated rows skipped."""

    records: tuple[Bioactivity, ...]
    total_count: int
    duplicate_count: int = 0

    @property
    def truncated(self) -> bool:
        return len(self.records) < self.total_count


@dataclass(frozen=True)
class ChEMBLProfile:
    molecule: ChEMBLMolecule
    activities: CompoundActivities


@dataclass(frozen=True)
class ChEMBLDocument:
    id: str
    doi: str | None
    pmid: int | None
    patent_id: str | None = None


class ChEMBLClient(ServiceClient):
    def __init__(
        self,
        http: HttpClient | None = None,
        *,
        requests_per_second: float = DEFAULT_INTEGRATIONS.chembl_requests_per_second,
    ) -> None:
        super().__init__(BASE_URL, http, requests_per_second=requests_per_second)

    async def document(self, document_id: str) -> ChEMBLDocument | None:
        """Find a paper or patent linked to a ChEMBL document ID."""
        document_id = _normalize_chembl_id(document_id)
        try:
            payload = await self._http.request_json(
                "GET", f"document/{document_id}.json", cache=True
            )
        except ServiceError as exc:
            if exc.status_code == 404:
                return None
            raise
        row = json_object(payload, "a ChEMBL document")
        if row.get("document_chembl_id") != document_id:
            raise ResponseError("ChEMBL returned a different document.")
        match row.get("doc_type"):
            case "PUBLICATION":
                return ChEMBLDocument(
                    document_id, text(row, "doi"), integer(row, "pubmed_id")
                )
            case "PATENT":
                patent_id = publication_number(text(row, "patent_id") or "")
                return (
                    ChEMBLDocument(document_id, None, None, patent_id)
                    if patent_id
                    else None
                )
            case _:
                return None

    async def molecule(self, chembl_id: str) -> ChEMBLMolecule | None:
        """Read one molecule, or None when ChEMBL has no such record."""
        chembl_id = _normalize_chembl_id(chembl_id)
        try:
            payload = await self._http.request_json(
                "GET", f"molecule/{chembl_id}.json", cache=True
            )
        except ServiceError as exc:
            if exc.status_code == 404:
                return None
            raise
        molecule = _parse_molecule(json_object(payload, "a ChEMBL molecule"))
        if molecule.chembl_id != chembl_id:
            raise ResponseError(
                f"asked ChEMBL for {chembl_id} and got {molecule.chembl_id}"
            )
        return molecule

    async def find_molecules(
        self,
        *,
        smiles: str | None = None,
        inchikey: str | None = None,
        max_records: int = 100,
    ) -> tuple[ChEMBLMolecule, ...]:
        """Find up to max_records entries for a SMILES or standard InChIKey."""
        check_integer(max_records, "max_records")
        if (smiles is None) == (inchikey is None):
            raise ParseError("give exactly one of smiles or inchikey")
        key = normalize_inchikey(
            inchi.MolToInchiKey(to_mol(smiles)) if smiles is not None else str(inchikey)
        )
        params = {
            "molecule_structures__standard_inchi_key": key,
            "only": _MOLECULE_FIELDS,
            "order_by": "molecule_chembl_id",
        }
        out: list[ChEMBLMolecule] = []
        async for rows, _ in self._pages(
            "molecule", "molecules", params, page_size=100, max_records=max_records
        ):
            molecules = [_parse_molecule(row) for row in rows]
            if any(molecule.inchikey != key for molecule in molecules):
                raise ResponseError(
                    "ChEMBL returned a molecule with a different InChIKey"
                )
            out.extend(molecules)
        return tuple(out)

    async def find_similar(
        self, smiles: str, *, threshold: int = 80, max_records: int = 50
    ) -> tuple[ChEMBLMolecule, ...]:
        """Find molecules similar to a SMILES in ChEMBL.

        Args:
            smiles: The query structure.
            threshold: Minimum ChEMBL Tanimoto score on a 0–100 scale.
            max_records: Maximum number of hits to read.

        Returns:
            Hits in ChEMBL order, which may include the query structure.
        """
        check_integer(threshold, "threshold", minimum=0, maximum=100)
        check_integer(max_records, "max_records")
        path = ("similarity", canonicalize_smiles(smiles), str(threshold))
        out = []
        try:
            async for rows, _ in self._pages(
                path,
                "molecules",
                {"only": _MOLECULE_FIELDS},
                page_size=100,
                max_records=max_records,
            ):
                out.extend(_parse_molecule(row) for row in rows)
        except ServiceError as exc:
            if exc.status_code != 404 or out:
                raise
        return tuple(out)

    async def activities(
        self,
        chembl_id: str,
        *,
        target_ids: Sequence[str] = (),
        activity_types: Sequence[str] = ("IC50", "EC50", "Ki", "Kd"),
        organism: str = "Homo sapiens",
        assay_types: Sequence[str] = ("B", "F"),
        units: str = "nM",
        max_records: int = 1000,
        page_size: int = 100,
    ) -> CompoundActivities:
        """Read a parent compound's measurements, including its salt forms.

        Use the parent ID for chembl_id. Empty target_ids includes all targets.
        Assay type B is binding and F is functional. Records follow activity-ID
        order, with the first occurrence retained for repeated IDs.
        """
        check_integer(max_records, "max_records")
        check_integer(page_size, "page_size", maximum=MAX_PAGE_SIZE)
        chembl_id = _normalize_chembl_id(chembl_id)
        if any(
            isinstance(values, str)
            for values in (target_ids, activity_types, assay_types)
        ):
            raise ParseError(
                "target, activity, and assay filters must each be a sequence"
            )
        if not (activity_types and assay_types and organism and units):
            raise ParseError(
                "activity, assay, organism, and unit filters cannot be empty"
            )
        params = {
            "parent_molecule_chembl_id": chembl_id,
            "target_organism": organism,
            "assay_type__in": ",".join(dict.fromkeys(assay_types)),
            "standard_type__in": ",".join(dict.fromkeys(activity_types)),
            "standard_units": units,
            "only": _ACTIVITY_FIELDS,
            "order_by": "activity_id",
        }
        if target_ids:
            params["target_chembl_id__in"] = ",".join(
                dict.fromkeys(_normalize_chembl_id(value) for value in target_ids)
            )
        out: list[Bioactivity] = []
        seen: set[int] = set()
        total = 0
        duplicate_count = 0
        async for rows, count in self._pages(
            "activity",
            "activities",
            params,
            page_size=page_size,
            max_records=max_records,
        ):
            total = count
            for row in rows:
                activity = _parse_activity(row)
                if activity.parent_molecule_chembl_id != chembl_id:
                    raise ResponseError(
                        "ChEMBL returned an activity for another compound"
                    )
                if activity.activity_id in seen:
                    duplicate_count += 1
                    continue
                seen.add(activity.activity_id)
                out.append(activity)
        if duplicate_count:
            logger.warning(
                "ChEMBL repeated %d activities for %s while paging",
                duplicate_count,
                chembl_id,
            )
        return CompoundActivities(tuple(out), total, duplicate_count)

    async def profile(
        self, chembl_id: str, *, target_ids: Sequence[str] = (), max_records: int = 1000
    ) -> ChEMBLProfile | None:
        """Read a molecule and its parent's measurements, or None if missing."""
        check_integer(max_records, "max_records")
        molecule = await self.molecule(chembl_id)
        if molecule is None:
            return None
        activities = await self.activities(
            molecule.parent_chembl_id or molecule.chembl_id,
            target_ids=target_ids,
            max_records=max_records,
        )
        return ChEMBLProfile(molecule, activities)

    async def approved_oral_drugs(
        self, *, max_records: int = 1000, page_size: int = 100
    ) -> tuple[ChEMBLMolecule, ...]:
        """Read records using ChEMBL's phase-4 oral drug filters."""
        check_integer(max_records, "max_records")
        check_integer(page_size, "page_size", maximum=MAX_PAGE_SIZE)
        params = {
            "max_phase": 4,
            "molecule_type": "Small molecule",
            "oral": "true",
            "therapeutic_flag": "true",
            "withdrawn_flag": "false",
            "only": _MOLECULE_FIELDS,
            "order_by": "molecule_chembl_id",
        }
        out = []
        async for rows, _ in self._pages(
            "molecule",
            "molecules",
            params,
            page_size=page_size,
            max_records=max_records,
        ):
            for row in rows:
                molecule = _parse_molecule(row)
                if molecule.phase != "approved" or molecule.oral is not True:
                    raise ResponseError(
                        "ChEMBL returned a molecule outside the drug filters"
                    )
                out.append(molecule)
        return tuple(out)

    async def _pages(
        self,
        resource: str | tuple[str, ...],
        key: str,
        params: Mapping[str, str | int],
        *,
        page_size: int,
        max_records: int,
    ) -> AsyncIterator[tuple[list[dict[str, Any]], int]]:
        """Read pages up to the requested record limit.

        Yields:
            Each page's rows and ChEMBL's total count for the query.
        """
        offset = 0
        first_total: int | None = None
        path = (
            f"{resource}.json"
            if isinstance(resource, str)
            else (*resource[:-1], f"{resource[-1]}.json")
        )
        while True:
            limit = min(page_size, max_records - offset)
            payload = json_object(
                await self._http.request_json(
                    "GET",
                    path,
                    params={**params, "limit": limit, "offset": offset},
                    cache=True,
                ),
                "a ChEMBL result page",
            )
            rows = records(payload, key)
            meta = json_object(payload.get("page_meta"), "ChEMBL page_meta")
            total = integer(meta, "total_count")
            if total is None or total < 0 or integer(meta, "offset") != offset:
                raise ResponseError("invalid ChEMBL pagination metadata")
            if first_total is not None and first_total != total:
                raise ResponseError("the ChEMBL result count changed while paging")
            first_total = total
            if (
                len(rows) > limit
                or offset + len(rows) > total
                or (not rows and offset < total)
            ):
                raise ResponseError("a ChEMBL page disagrees with its record count")
            yield rows, total
            offset += len(rows)
            if offset >= total or offset >= max_records:
                return
            if meta.get("next") is None:
                raise ResponseError("ChEMBL stopped paging before the last record")


def _normalize_chembl_id(value: str) -> str:
    """Return a ChEMBL ID in upper case, rejecting anything else."""
    text_value = value.strip().upper()
    if not _ID_PATTERN.fullmatch(text_value):
        raise ParseError(f"invalid ChEMBL identifier: {value!r}")
    return text_value


def _phase(value: float | None) -> ClinicalPhase:
    if value not in _PHASES:
        logger.warning("Unknown ChEMBL max_phase: %r", value)
    return _PHASES.get(value, "unknown_phase")


def _parse_molecule(row: Mapping[str, Any]) -> ChEMBLMolecule:
    chembl_id = text(row, "molecule_chembl_id")
    if not chembl_id:
        raise ResponseError("ChEMBL molecule is missing its identifier")
    structures = json_object(
        row.get("molecule_structures") or {}, "molecule_structures"
    )
    hierarchy = json_object(row.get("molecule_hierarchy") or {}, "molecule_hierarchy")
    smiles = text(structures, "canonical_smiles")
    if smiles is not None:
        try:
            smiles = canonicalize_smiles(smiles)
        except ParseError as exc:
            raise ResponseError(f"cannot parse the structure of {chembl_id}") from exc
    return ChEMBLMolecule(
        chembl_id=chembl_id,
        name=text(row, "pref_name"),
        smiles=smiles,
        inchikey=text(structures, "standard_inchi_key"),
        parent_chembl_id=text(hierarchy, "parent_chembl_id"),
        phase=_phase(number(row, "max_phase")),
        therapeutic=flag(row, "therapeutic_flag"),
        oral=flag(row, "oral"),
        withdrawn=flag(row, "withdrawn_flag"),
    )


def _parse_activity(row: Mapping[str, Any]) -> Bioactivity:
    activity_id = integer(row, "activity_id")
    molecule_id = text(row, "molecule_chembl_id")
    if activity_id is None or activity_id < 1 or not molecule_id:
        raise ResponseError("ChEMBL activity is missing its identity")
    efficiency = json_object(row.get("ligand_efficiency") or {}, "ligand_efficiency")
    relation = text(row, "standard_relation")
    return Bioactivity(
        activity_id=activity_id,
        molecule_chembl_id=molecule_id,
        parent_molecule_chembl_id=text(row, "parent_molecule_chembl_id"),
        target_chembl_id=text(row, "target_chembl_id"),
        target_name=text(row, "target_pref_name"),
        organism=text(row, "target_organism"),
        standard_type=text(row, "standard_type"),
        standard_relation=cast(ResultRelation, relation)
        if relation in STANDARD_RELATIONS
        else None,
        standard_value=number(row, "standard_value"),
        standard_units=text(row, "standard_units"),
        pchembl_value=number(row, "pchembl_value"),
        assay_chembl_id=text(row, "assay_chembl_id"),
        assay_type=text(row, "assay_type"),
        assay_description=text(row, "assay_description"),
        validity_comment=text(row, "data_validity_comment"),
        potential_duplicate=flag(row, "potential_duplicate"),
        document_chembl_id=text(row, "document_chembl_id"),
        journal=text(row, "document_journal"),
        year=integer(row, "document_year"),
        ligand_efficiency=number(efficiency, "le"),
        lipophilic_efficiency=number(efficiency, "lle"),
        binding_efficiency_index=number(efficiency, "bei"),
    )
