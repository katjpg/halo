from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx
from rdkit import Chem, DataStructs

from halo.chem.fingerprints import Fingerprints
from halo.chem.mol import canonicalize_smiles, normalize_inchikey, to_mol, to_smiles
from halo.clients.http import (
    HttpClient,
    ResponseError,
    ServiceClient,
    ServiceError,
    check_integer,
    integer,
    json_object,
    number,
    records,
    text,
)
from halo.config.settings import DEFAULT_INTEGRATIONS
from halo.core.errors import ParseError
from halo.models.research import Publication
from halo.models.research import publication_number as publication_number

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest"
NOT_FOUND_CODES = frozenset({"PUGREST.NotFound", "PUGVIEW.NotFound"})
CID_BATCH_SIZE = 100
_PROPERTIES = (
    "SMILES,MolecularWeight,XLogP,InChIKey,TPSA,HBondDonorCount,"
    "HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,IUPACName"
)
Namespace = Literal["name", "smiles", "inchikey"]
CrossReference = Literal["PatentID", "PubMedID"]


@dataclass(frozen=True, slots=True)
class PubChemCompound:
    """PubChem properties for one CID, with the SMILES canonicalized locally.

    Mass is in g/mol and TPSA is in Å². LogP and hydrogen-bond counts use
    PubChem's XLogP3 and counting methods.
    """

    cid: int
    smiles: str
    inchikey: str | None
    molecular_weight: float | None
    xlogp: float | None
    tpsa: float | None
    h_bond_donors: int | None
    h_bond_acceptors: int | None
    rotatable_bonds: int | None
    heavy_atom_count: int | None
    iupac_name: str | None
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SimilarityHit:
    compound: PubChemCompound
    morgan_tanimoto: float


@dataclass(frozen=True)
class SimilarityResult:
    """PubChem neighbors in descending local Morgan Tanimoto order.

    Threshold uses PubChem's 0–100 scale. Retrieved count is before local filtering.
    """

    query_smiles: str
    threshold: int
    retrieved_count: int
    hits: tuple[SimilarityHit, ...]
    fingerprint_method: str


class PubChemClient(ServiceClient):
    """A PUG-REST client that backs off when PubChem reports high load."""

    def __init__(
        self,
        http: HttpClient | None = None,
        *,
        requests_per_second: float = DEFAULT_INTEGRATIONS.pubchem_requests_per_second,
    ) -> None:
        super().__init__(BASE_URL, http, requests_per_second=requests_per_second)

    async def cids(
        self, identifier: str, *, namespace: Namespace = "name"
    ) -> tuple[int, ...]:
        """Resolve an identifier to CIDs in PubChem's order. No match returns ()."""
        value = identifier.strip()
        if not value:
            raise ParseError("identifier cannot be empty")
        if namespace not in ("name", "smiles", "inchikey"):
            raise ParseError("unknown PubChem identifier namespace")
        if namespace == "smiles":
            value = canonicalize_smiles(value)
        elif namespace == "inchikey":
            value = normalize_inchikey(value)
        payload = await self._lookup(
            "POST", f"pug/compound/{namespace}/cids/JSON", data={namespace: value}
        )
        return () if payload is None else _parse_cids(payload)

    async def compounds(
        self, cids: Sequence[int], *, batch_size: int = CID_BATCH_SIZE
    ) -> tuple[PubChemCompound, ...]:
        """Read one compound per distinct CID, in input order."""
        check_integer(batch_size, "batch_size")
        for cid in cids:
            check_integer(cid, "cid")
        ids = list(dict.fromkeys(cids))
        out: list[PubChemCompound] = []
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            payload = json_object(
                await self._http.request_json(
                    "POST",
                    f"pug/compound/cid/property/{_PROPERTIES}/JSON",
                    data={"cid": ",".join(map(str, batch))},
                    replayable=True,
                    cache=True,
                    response_hook=self._throttle,
                ),
                "a PubChem property table",
            )
            table = json_object(payload.get("PropertyTable"), "a PubChem PropertyTable")
            by_cid: dict[int, PubChemCompound] = {}
            for row in records(table, "Properties"):
                compound = _parse_compound(row)
                if compound.cid in by_cid:
                    raise ResponseError(f"PubChem repeated CID {compound.cid}")
                by_cid[compound.cid] = compound
            if set(by_cid) != set(batch):
                raise ResponseError(
                    "PubChem properties do not match the CIDs asked for"
                )
            out.extend(by_cid[cid] for cid in batch)
        return tuple(out)

    async def synonyms(
        self, cids: Sequence[int], *, limit: int = 5, batch_size: int = CID_BATCH_SIZE
    ) -> dict[int, tuple[str, ...]]:
        """Read synonyms for every requested CID, using () for none found."""
        check_integer(limit, "limit")
        check_integer(batch_size, "batch_size")
        for cid in cids:
            check_integer(cid, "cid")
        ids = list(dict.fromkeys(cids))
        out: dict[int, tuple[str, ...]] = dict.fromkeys(ids, ())
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            payload = await self._lookup(
                "POST",
                "pug/compound/cid/synonyms/JSON",
                data={"cid": ",".join(map(str, batch))},
            )
            if payload is None:
                continue
            info = json_object(
                payload.get("InformationList"), "a PubChem InformationList"
            )
            expected = set(batch)
            seen: set[int] = set()
            for row in records(info, "Information"):
                cid = integer(row, "CID")
                if cid is None or cid not in expected or cid in seen:
                    raise ResponseError(
                        "PubChem synonyms name a CID that was not asked for"
                    )
                seen.add(cid)
                names = row.get("Synonym", [])
                if not isinstance(names, list) or any(
                    not isinstance(s, str) for s in names
                ):
                    raise ResponseError("expected a list of PubChem synonyms")
                out[cid] = tuple(dict.fromkeys(names))[:limit]
        return out

    async def cross_references(
        self, cids: Sequence[int], field: CrossReference
    ) -> tuple[tuple[int, str], ...]:
        """Read publication IDs with the CID associated with each link."""
        if field not in ("PatentID", "PubMedID"):
            raise ParseError("Unknown PubChem cross-reference field.")
        for cid in cids:
            check_integer(cid, "cid")
        requested = sorted(set(cids))
        links = set()
        for start in range(0, len(requested), CID_BATCH_SIZE):
            batch = requested[start : start + CID_BATCH_SIZE]
            path = ",".join(map(str, batch))
            payload = await self._lookup(
                "GET", f"pug/compound/cid/{path}/xrefs/{field}/JSON"
            )
            if payload is None:
                continue
            info = json_object(
                payload.get("InformationList"), "PubChem cross-references"
            )
            expected = set(batch)
            for row in records(info, "Information"):
                cid = integer(row, "CID")
                ids = row.get(field, [])
                if cid is None or cid not in expected or not isinstance(ids, list):
                    raise ResponseError("PubChem returned invalid cross-references.")
                for value in ids:
                    if field == "PatentID" and isinstance(value, str):
                        ref = publication_number(value)
                    elif field == "PubMedID" and type(value) is int and value > 0:
                        ref = str(value)
                    else:
                        ref = None
                    if ref:
                        links.add((cid, ref))
        return tuple(sorted(links))

    async def patent_abstract(self, publication_id: str) -> Publication | None:
        """Read the abstract of a verified patent publication."""
        publication_id = publication_number(publication_id)
        if publication_id is None:
            raise ParseError("Invalid patent publication number.")
        payload = await self._lookup(
            "GET",
            f"pug_view/data/patent/{publication_id}/JSON",
            params={"heading": "Abstract"},
        )
        if payload is None:
            return None
        row = json_object(payload.get("Record"), "PubChem patent")
        if publication_number(text(row, "RecordAccession") or "") != publication_id:
            raise ResponseError("PubChem returned a different patent publication.")
        title = " ".join((text(row, "RecordTitle") or "").split())
        sections = records(row, "Section", default=[])
        abstract = " ".join(
            word
            for section in sections
            if section.get("TOCHeading") == "Abstract"
            for info in records(section, "Information")
            for part in records(
                json_object(info.get("Value"), "patent abstract"), "StringWithMarkup"
            )
            for word in (text(part, "String") or "").split()
        )
        if not title or not abstract:
            return None
        url_id = publication_id.replace("-", "")
        return Publication(
            publication_id,
            title,
            abstract,
            f"https://patents.google.com/patent/{url_id}/en",
        )

    async def find_similar(
        self,
        smiles: str,
        *,
        threshold: int = 90,
        max_records: int = 100,
        top_n: int = 25,
        include_synonyms: bool = False,
        max_synonyms: int = 5,
        exclude_self: bool = True,
        single_component: bool = True,
    ) -> SimilarityResult:
        """Find PubChem neighbors and rank them by Morgan Tanimoto.

        Args:
            smiles: The query structure.
            threshold: Minimum PubChem similarity on a 0–100 scale.
            max_records: Maximum number of hits to fetch.
            top_n: Maximum number of neighbors to return.
            include_synonyms: Fetch synonyms for the retained neighbors.
            max_synonyms: Maximum number of synonyms per neighbor.
            exclude_self: Exclude the exact query structure.
            single_component: Require connected molecules in the query and hits.

        Returns:
            Neighbors ordered by descending local Morgan Tanimoto score.
            Lower CIDs win when structures repeat or scores tie.
        """
        check_integer(threshold, "threshold", minimum=0, maximum=100)
        check_integer(max_records, "max_records")
        check_integer(top_n, "top_n")
        check_integer(max_synonyms, "max_synonyms")
        if top_n > max_records:
            raise ParseError("top_n cannot exceed max_records")
        mol = to_mol(smiles)
        if single_component and len(Chem.GetMolFrags(mol)) > 1:
            raise ParseError("query must be a single connected molecule")
        canonical = to_smiles(mol)
        payload = await self._lookup(
            "POST",
            "pug/compound/fastsimilarity_2d/smiles/cids/JSON",
            params={"Threshold": threshold, "MaxRecords": max_records},
            data={"smiles": canonical},
        )
        ids = () if payload is None else _parse_cids(payload)
        if len(ids) > max_records:
            raise ResponseError("PubChem returned more CIDs than were asked for")
        kept: dict[str, PubChemCompound] = {}
        for compound in await self.compounds(ids):
            if exclude_self and compound.smiles == canonical:
                continue
            if single_component and "." in compound.smiles:
                continue
            previous = kept.get(compound.smiles)
            if previous is None or compound.cid < previous.cid:
                kept[compound.smiles] = compound
        compounds = list(kept.values())
        fps = Fingerprints.from_mols([mol, *(to_mol(c.smiles) for c in compounds)])
        scores = DataStructs.BulkTanimotoSimilarity(fps.bits[0], fps.bits[1:])
        hits = sorted(
            (
                SimilarityHit(compound, float(score))
                for compound, score in zip(compounds, scores, strict=True)
            ),
            key=lambda hit: (-hit.morgan_tanimoto, hit.compound.cid),
        )[:top_n]
        if include_synonyms and hits:
            names = await self.synonyms(
                [hit.compound.cid for hit in hits], limit=max_synonyms
            )
            hits = [
                replace(
                    hit,
                    compound=replace(hit.compound, synonyms=names[hit.compound.cid]),
                )
                for hit in hits
            ]
        return SimilarityResult(canonical, threshold, len(ids), tuple(hits), fps.method)

    def _throttle(self, response: httpx.Response) -> None:
        """Slow later requests when PubChem reports its service load as busy."""
        status = response.headers.get("X-Throttling-Control", "").lower()
        for colour, pause_s in (("black", 60.0), ("red", 2.0), ("yellow", 0.5)):
            if colour in status:
                self._http.limiter.defer(pause_s)
                return

    async def _lookup(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Send a read request, returning None when PubChem has no such record."""
        try:
            payload = await self._http.request_json(
                method,
                path,
                params=params,
                data=data,
                replayable=True,
                cache=True,
                response_hook=self._throttle,
            )
        except ServiceError as exc:
            if exc.status_code == 404 and _fault_code(exc.payload) in NOT_FOUND_CODES:
                return None
            raise
        return json_object(payload, "a PubChem record")


def _fault_code(payload: Any) -> str | None:
    """Read the PUG-REST or PUG-View error code from a Fault body."""
    if not isinstance(payload, dict):
        return None
    fault = payload.get("Fault")
    code = fault.get("Code") if isinstance(fault, dict) else None
    return code if isinstance(code, str) else None


def _parse_cids(payload: Mapping[str, Any]) -> tuple[int, ...]:
    if "Waiting" in payload:
        raise ResponseError(
            "PubChem queued the search instead of answering it; retry with fewer max_records"
        )
    identifiers = json_object(payload.get("IdentifierList"), "a PubChem IdentifierList")
    ids = identifiers.get("CID")
    if not isinstance(ids, list) or any(type(cid) is not int or cid < 1 for cid in ids):
        raise ResponseError("expected a list of PubChem CIDs")
    return tuple(dict.fromkeys(ids))


def _parse_compound(row: Mapping[str, Any]) -> PubChemCompound:
    cid = integer(row, "CID")
    if cid is None or cid < 1:
        raise ResponseError("PubChem compound is missing its CID")
    smiles = text(row, "SMILES") or text(row, "IsomericSMILES")
    if not smiles:
        raise ResponseError(f"PubChem CID {cid} has no SMILES")
    try:
        canonical = canonicalize_smiles(smiles)
    except ParseError as exc:
        raise ResponseError(f"cannot parse the structure of PubChem CID {cid}") from exc
    return PubChemCompound(
        cid=cid,
        smiles=canonical,
        inchikey=text(row, "InChIKey"),
        molecular_weight=number(row, "MolecularWeight"),
        xlogp=number(row, "XLogP"),
        tpsa=number(row, "TPSA"),
        h_bond_donors=integer(row, "HBondDonorCount"),
        h_bond_acceptors=integer(row, "HBondAcceptorCount"),
        rotatable_bonds=integer(row, "RotatableBondCount"),
        heavy_atom_count=integer(row, "HeavyAtomCount"),
        iupac_name=text(row, "IUPACName"),
    )
