import asyncio
import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import zip_longest
from math import isfinite
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar
from unicodedata import normalize

from rdkit import DataStructs
from rdkit.Chem import inchi

from halo.chem.fingerprints import Fingerprints
from halo.chem.mol import canonicalize_smiles, to_mol
from halo.core.errors import LimitError, ProviderError
from halo.llm.agents.researcher import (
    SearchPlan,
    build_context,
    plan_search,
    research_context,
    respond,
)
from halo.llm.runtime import ModelRunner, ModelTrace, format_request
from halo.llm.validate import ValidationIssue
from halo.models.analysis import ResultValue
from halo.models.research import (
    Article,
    Citation,
    EvidenceScope,
    FullText,
    Passage,
    Publication,
    Response,
    SourceExclusion,
    SourceLink,
    SourceSearch,
)
from halo.refs import chembl_activity_ref, passage_ref
from halo.types import MoleculeId

if TYPE_CHECKING:
    from halo.clients.chem.chembl import (
        Bioactivity,
        ChEMBLDocument,
        ChEMBLMolecule,
        CompoundActivities,
    )
    from halo.clients.chem.pubchem import PubChemCompound, SimilarityResult


MAX_WORD_COUNT = 24
RESEARCH_TIMEOUT_S = 130
MAX_DOCUMENTS = 5
MAX_PATENTS = 8
MAX_NEIGHBORS = 5
LOOKUP_TIMEOUT_S = 18
DOI = re.compile(r"10\.\d{4,9}/[-._;()/a-z0-9]+", re.IGNORECASE)
CHEMBL_ID = re.compile(r"\bchembl[1-9]\d*\b")
TARGET_SYMBOL = re.compile(
    r"\b(?:[a-z]*[A-Z]{2,}[A-Za-z0-9]*|[A-Za-z]+\d+[A-Za-z0-9]*)\b"
)
ENDPOINT = re.compile(r"\b(?:p?(?:ac|cc|ec|gi|ic)\s*50|p?k[di])\b")
ASSAY_TYPES = {"B": "binding", "F": "functional"}
ASSAY_PATTERNS = {
    "kinase": r"\b(?:kinase (?:assays?|activit\w*|inhibit\w*)|inhibition of (?:\w+ ){0,4}kinase)\b",
    "binding": r"\b(?:binding|binds|binders?)\b",
    "functional": r"\bfunctional\b",
    "biochemical": r"\bbiochemical\b",
    "cellular": r"\b(?:cellular|cell based)\b",
    "spr": r"\b(?:spr|surface plasmon resonance)\b",
    "patch clamp": r"\bpatch clamp\b",
    "radioligand": r"\bradioligand\b",
    "elisa": r"\belisa\b",
}
STRUCTURE_SCOPES: tuple[EvidenceScope, ...] = ("exact", "analog")
STOP_WORDS = frozenset(
    "and are about assay assays can could does find for from have how its "
    "molecule molecules our paper papers patent patents show source sources "
    "structure the their this use was were what which with would".split()
)
T = TypeVar("T")
PublicationT = TypeVar("PublicationT", bound=Publication)


class ChemicalLiterature(Protocol):
    async def find_molecules(
        self, *, smiles: str, max_records: int
    ) -> tuple["ChEMBLMolecule", ...]: ...
    async def find_similar(
        self, smiles: str, *, threshold: int, max_records: int
    ) -> tuple["ChEMBLMolecule", ...]: ...
    async def activities(
        self, chembl_id: str, *, max_records: int
    ) -> "CompoundActivities": ...
    async def document(self, document_id: str) -> "ChEMBLDocument | None": ...


class CompoundIndex(Protocol):
    async def cids(
        self, identifier: str, *, namespace: Literal["inchikey"]
    ) -> tuple[int, ...]: ...
    async def compounds(self, cids: Sequence[int]) -> tuple["PubChemCompound", ...]: ...
    async def synonyms(
        self, cids: Sequence[int], *, limit: int
    ) -> dict[int, tuple[str, ...]]: ...
    async def cross_references(
        self, cids: Sequence[int], field: Literal["PatentID", "PubMedID"]
    ) -> tuple[tuple[int, str], ...]: ...
    async def find_similar(
        self,
        smiles: str,
        *,
        threshold: int,
        max_records: int,
        top_n: int,
        single_component: bool,
    ) -> "SimilarityResult": ...
    async def patent_abstract(self, publication_id: str) -> Publication | None: ...


class PaperIndex(Protocol):
    async def search(self, query: str, *, limit: int = 12) -> tuple["Article", ...]: ...
    async def search_full_text(
        self, query: str, *, limit: int = 5
    ) -> tuple["Article", ...]: ...
    async def full_text(self, pmcid: str) -> "FullText": ...


def terms(value: str) -> tuple[str, ...]:
    """Extract distinct terms for ranking and topic search."""
    return tuple(
        dict.fromkeys(
            word
            for word in re.findall(r"\b(?:[a-z][a-z0-9]{2,}|ki|kd)\b", value.casefold())
            if word not in STOP_WORDS
        )
    )


def select_excerpt(
    abstract: str, question: str, aliases: tuple[str, ...] = ()
) -> tuple[int, str]:
    """Pick a relevant excerpt of up to MAX_WORD_COUNT words from one sentence."""
    query_terms = set(terms(question))
    names = tuple(re.sub(r"[^a-z0-9]", "", name.casefold()) for name in aliases)
    best_score = (-1, False, -1)
    best = (1, "")
    for idx, sentence in enumerate(
        re.split(r"(?<=[.!?])\s+(?=[A-Z(])", abstract), start=1
    ):
        words = sentence.split()
        for start in range(max(1, len(words) - MAX_WORD_COUNT + 1)):
            end = start + MAX_WORD_COUNT
            window = " ".join(words[start:end])
            overlap = len(query_terms.intersection(terms(window)))
            compact = re.sub(r"[^a-z0-9]", "", window.casefold())
            score = (
                int(bool(overlap)),
                any(name and name in compact for name in names),
                overlap,
            )
            if score > best_score:
                best_score = score
                best = (
                    idx,
                    ("…" if start else "") + window + ("…" if end < len(words) else ""),
                )
    return best


def select_document(
    records: Sequence[tuple[PublicationT, SourceLink]], question: str
) -> tuple[PublicationT, SourceLink] | None:
    query_terms = set(terms(question))
    return max(
        records,
        key=lambda pair: (
            3 * len(query_terms.intersection(terms(pair[0].title)))
            + len(query_terms.intersection(terms(pair[0].abstract)))
        ),
        default=None,
    )


def split_sentences(text: str) -> Iterator[str]:
    """Split sentences while preserving text and common scientific abbreviations."""
    start = 0
    for match in re.finditer(r"[.!?][\"'”’)\]]*(?:\[\d+(?:[-–,\s]+\d+)*\])*\s+", text):
        prefix = text[start : match.start() + 1]
        if re.search(
            r"(?:\b(?:figs?|eqs?|refs?|dr|prof|vs|approx|ca|al)\.|\be\.g\.|\bi\.e\.|\b(?:[A-Z]\.){2,})$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        sentence = text[start : match.end()].strip()
        if sentence:
            yield sentence
        start = match.end()
    tail = text[start:].strip()
    if re.search(r"[.!?][\"'”’)\]]*(?:\[\d+(?:[-–,\s]+\d+)*\])*$", tail):
        yield tail


def extract_passages(
    article: "Article", document: "FullText", plan: SearchPlan, question: str
) -> tuple[Passage, ...]:
    """Select sentences that mention both a structural change and a property."""
    change_terms = tuple(
        dict.fromkeys(_normalized(value) for value in plan.change_terms)
    )
    property_terms = tuple(
        dict.fromkeys(_normalized(value) for value in plan.property_terms)
    )
    criteria = _SourceQuery.parse(question, ())
    ranked = []
    seen = set()
    for paragraph_index, paragraph in enumerate(document.paragraphs, start=1):
        for sentence_index, sentence in enumerate(
            split_sentences(paragraph.text), start=1
        ):
            text = _normalized(sentence)
            changes = sum(bool(term) and _mentions(text, term) for term in change_terms)
            properties = sum(
                bool(term) and _mentions(text, term) for term in property_terms
            )
            if not changes or not properties or text in seen:
                continue
            if not criteria.matches_document(
                Publication(article.id, "", sentence, article.url)
            ):
                continue
            seen.add(text)
            passage = Passage(
                id=passage_ref(
                    "paper",
                    document.pmcid,
                    sentence_index,
                    paragraph_index=paragraph_index,
                ),
                publication_id=document.pmcid,
                source_type="paper",
                title=document.title or article.title,
                text=sentence,
                url=f"https://europepmc.org/articles/{document.pmcid}",
                link=SourceLink("topic"),
                pmcid=document.pmcid,
                doi=document.doi or article.doi or None,
                year=document.year or article.year,
                license_url=document.license_url,
                locator=f"{paragraph.locator}/sentence-{sentence_index}",
            )
            ranked.append(
                (-(changes + properties), paragraph_index, sentence_index, passage)
            )
    return tuple(item[-1] for item in sorted(ranked, key=lambda item: item[:3])[:3])


@dataclass(frozen=True)
class _Neighbor:
    smiles: str
    chembl: tuple["ChEMBLMolecule", ...]
    cids: tuple[int, ...]
    similarity: float
    fingerprint_method: str


@dataclass(frozen=True)
class _Match:
    link: SourceLink
    activity: "Bioactivity | None" = None


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", normalize("NFKC", text).casefold()))


def _mentions(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _symbols(text: str) -> set[str]:
    return {word.casefold() for word in TARGET_SYMBOL.findall(normalize("NFKC", text))}


def _target_names(name: str) -> set[str]:
    words = _normalized(name).split()
    if not words:
        return set()
    symbol = _symbols(name.split()[-1])
    names = symbol | {" ".join(words)}
    if symbol:
        words = words[:-1] if len(words) >= 5 else []
    if len(words) >= 3:
        names.add("".join(word[0] for word in words))
    return names


def _assays(text: str) -> set[str]:
    return {
        name for name, pattern in ASSAY_PATTERNS.items() if re.search(pattern, text)
    }


def _endpoints(text: str) -> set[str]:
    return {endpoint.replace(" ", "") for endpoint in ENDPOINT.findall(text)}


@dataclass
class _SourceQuery:
    targets: set[str]
    assays: set[str]
    endpoints: set[str]
    ids: set[str]
    requires_assay: bool

    @classmethod
    def parse(
        cls,
        question: str,
        activities: Sequence["Bioactivity"],
        aliases: Sequence[str] = (),
    ) -> "_SourceQuery":
        text = _normalized(question)
        ids = set(CHEMBL_ID.findall(text))
        endpoints = _endpoints(text)
        assays = _assays(text)
        other = (
            STOP_WORDS
            | set(" ".join(ASSAY_PATTERNS).split())
            | set(
                "activity adme admet analogs cell chembl compound data fda halo iupac "
                "lead list mmp nci pdb plasmon pubchem report resonance sar smiles cc by xml doi pmcid pmid "
                "surface target us value values wo".split()
            )
            | {_normalized(name) for name in aliases}
        )
        targets = _symbols(question) - other - ids - endpoints
        targets -= {endpoint.removesuffix("50") for endpoint in endpoints}
        names = [
            _target_names(name)
            for name in {row.target_name for row in activities if row.target_name}
        ]
        for group in names:
            targets.update(name for name in group - other if _mentions(text, name))
        if not targets and not ids:
            subject = re.search(
                r"\b(?:about|against|for|on|target(?:ing)?) (?:the )?([a-z][a-z0-9 ]*?)"
                r"(?= (?:activity|and|assays?|binding|inhibition|kinase inhibition|potency|using)\b|$)",
                " ".join(ENDPOINT.sub("", text).split()),
            )
            if subject is None and endpoints:
                subject = re.match(
                    rf"(?:(?:find|show|look up|report|list) )?([a-z][a-z0-9]+) (?={ENDPOINT.pattern})",
                    text,
                )
            if subject:
                name = subject[1].strip()
                if name not in other and (
                    re.fullmatch(r"[a-z][a-z0-9]+", name)
                    or re.search(
                        r"\b(?:channel|kinase|protease|receptor|transporter)\b", name
                    )
                ):
                    targets.add(name)
        targets |= set().union(*(group for group in names if group & targets))
        return cls(
            targets, assays, endpoints, ids, bool(re.search(r"\bassays?\b", text))
        )

    def matches_activity(self, activity: "Bioactivity") -> bool:
        ids = {
            (activity.target_chembl_id or "").casefold(),
            (activity.assay_chembl_id or "").casefold(),
        }
        assays = _assays(_normalized(activity.assay_description or ""))
        assay_type = ASSAY_TYPES.get(activity.assay_type or "")
        if assay_type:
            assays.difference_update(("binding", "functional"))
            assays.add(assay_type)
        return (
            (
                not self.targets
                or bool(self.targets & _target_names(activity.target_name or ""))
            )
            and self.ids <= ids
            and self.assays <= assays
            and (
                not self.endpoints
                or bool(
                    self.endpoints
                    & _endpoints(_normalized(activity.standard_type or ""))
                )
            )
        )

    def matches_document(self, document: Publication) -> bool:
        """Match target and assay details within one part of the abstract or title."""
        clauses = re.split(
            r"(?<=[.!?;])\s+|\s+(?:but|whereas|while)\s+",
            document.abstract,
            flags=re.IGNORECASE,
        )
        for span in (document.title, *clauses):
            text = _normalized(span)
            endpoints = _endpoints(text)
            if (
                (
                    not self.targets
                    or any(_mentions(text, name) for name in self.targets)
                )
                and self.ids <= set(CHEMBL_ID.findall(text))
                and self.assays <= _assays(text)
                and (not self.endpoints or bool(self.endpoints & endpoints))
                and (
                    not self.requires_assay
                    or endpoints
                    or re.search(r"\bassays?\b", text)
                )
            ):
                return True
        return False


def _relevance(
    match: _Match,
    query_terms: set[str],
    document: Publication | None = None,
) -> tuple[int, int, int, float]:
    """Rank eligible sources by query overlap, then by similarity."""
    activity = match.activity
    title_terms = set(terms(document.title)) if document else set()
    abstract_terms = set(terms(document.abstract)) if document else set()
    metadata_terms: set[str] = set()
    if activity:
        assay_type = ASSAY_TYPES.get(activity.assay_type or "", "")
        metadata_terms = _target_names(activity.target_name or "") | set(
            terms(
                f"{activity.target_chembl_id or ''} {activity.assay_chembl_id or ''} "
                f"{activity.standard_type or ''} {assay_type} {activity.assay_description or ''}"
            )
        )
    source_terms = metadata_terms | title_terms | abstract_terms
    return (
        len(query_terms & source_terms),
        len(query_terms & metadata_terms),
        3 * len(query_terms & title_terms) + len(query_terms & abstract_terms),
        match.link.morgan_tanimoto or 0.0,
    )


def _pubchem_references(
    rows: Sequence[tuple[int, str]], compounds: Mapping[int, SourceLink], *, limit: int
) -> dict[str, list[_Match]]:
    """Budget references per scope, taking one per compound before taking more."""
    by_cid: dict[int, list[str]] = defaultdict(list)
    for cid, ref in rows:
        if cid in compounds:
            by_cid[cid].append(ref)
    matches: dict[str, list[_Match]] = defaultdict(list)
    for scope in STRUCTURE_SCOPES:
        cids = [
            cid
            for cid, link in compounds.items()
            if link.scope == scope and cid in by_cid
        ]
        chosen = set(
            list(
                dict.fromkeys(
                    ref
                    for group in zip_longest(*(by_cid[cid] for cid in cids))
                    for ref in group
                    if ref is not None
                )
            )[:limit]
        )
        for cid in cids:
            for ref in by_cid[cid]:
                if ref in chosen:
                    matches[ref].append(_Match(compounds[cid]))
    return matches


async def lookup(name: str, request: Awaitable[T], unavailable: list[str]) -> T | None:
    try:
        async with asyncio.timeout(LOOKUP_TIMEOUT_S):
            return await request
    except (ProviderError, TimeoutError):
        unavailable.append(name)
        return None


class SourceFinder:
    """Literature and patent search with structure links."""

    def __init__(
        self, chembl: ChemicalLiterature, pubchem: CompoundIndex, papers: PaperIndex
    ) -> None:
        self.chembl = chembl
        self.pubchem = pubchem
        self.papers = papers

    async def find_passages(self, plan: SearchPlan, question: str) -> SourceSearch:
        """Retrieve license-verified full-text passages for a focused query."""
        unavailable: list[str] = []
        exclusions: list[SourceExclusion] = []
        rows = await lookup(
            "Europe PMC full-text search",
            self.papers.search_full_text(plan.query, limit=MAX_DOCUMENTS),
            unavailable,
        )
        selected = {}
        for article in rows or ():
            if article.pmcid is None:
                exclusions.append(
                    SourceExclusion(article.id, "The record has no PMCID.")
                )
            else:
                selected.setdefault(article.pmcid, article)
        articles = tuple(selected.values())[:MAX_DOCUMENTS]
        documents = await asyncio.gather(
            *(
                lookup(
                    f"Europe PMC full text {article.pmcid}",
                    self.papers.full_text(article.pmcid),
                    unavailable,
                )
                for article in articles
            )
        )
        passages = []
        seen = set()
        for article, document in zip(articles, documents, strict=True):
            if document is None:
                continue
            if document.license_url is None:
                exclusions.append(
                    SourceExclusion(
                        article.id,
                        "An article-level CC BY license link was not confirmed.",
                    )
                )
                continue
            if not re.fullmatch(
                r"https://creativecommons\.org/licenses/by/(?:1\.0|2\.0|2\.5|3\.0|4\.0)/",
                document.license_url,
            ):
                exclusions.append(
                    SourceExclusion(article.id, "The article license is not CC BY.")
                )
                continue
            found = extract_passages(article, document, plan, question)
            if not found:
                exclusions.append(
                    SourceExclusion(
                        article.id,
                        "No sentence matched the change, property, and requested target or assay.",
                    )
                )
            for passage in found:
                key = _normalized(passage.text)
                if key in seen:
                    exclusions.append(
                        SourceExclusion(
                            article.id, "The passage duplicates an earlier result."
                        )
                    )
                    continue
                seen.add(key)
                passages.append(passage)
        return SourceSearch(
            tuple(passages), tuple(unavailable), tuple(dict.fromkeys(exclusions))
        )

    async def exact_activities(
        self, smiles: str, *, question: str | None = None
    ) -> tuple["Bioactivity", ...]:
        """Read exact-structure activities, optionally filtered by target and assay."""
        molecules = await self._exact_chembl(canonicalize_smiles(smiles))
        rows = await self._read_activities(molecules)
        if question is None:
            return rows
        names = tuple(mol.name for mol in molecules if mol.name)
        criteria = _SourceQuery.parse(question, rows, names)
        return tuple(row for row in rows if criteria.matches_activity(row))

    async def _exact_chembl(self, smiles: str) -> tuple["ChEMBLMolecule", ...]:
        molecules = await self.chembl.find_molecules(smiles=smiles, max_records=4)
        return tuple(molecule for molecule in molecules if molecule.smiles == smiles)

    async def _read_activities(
        self,
        molecules: Sequence["ChEMBLMolecule"],
        unavailable: list[str] | None = None,
    ) -> tuple["Bioactivity", ...]:
        matched_ids = {mol.chembl_id for mol in molecules}
        parents = sorted({mol.parent_chembl_id or mol.chembl_id for mol in molecules})

        async def read(parent: str) -> "CompoundActivities | None":
            request = self.chembl.activities(parent, max_records=100)
            if unavailable is None:
                return await request
            return await lookup(f"ChEMBL activities {parent}", request, unavailable)

        responses = await asyncio.gather(*(read(parent) for parent in parents))
        return tuple(
            row
            for response in responses
            if response is not None
            for row in response.records
            if row.molecule_chembl_id in matched_ids
        )

    async def exact_cids(self, smiles: str) -> tuple[int, ...]:
        key = inchi.MolToInchiKey(to_mol(smiles))
        ids = await self.pubchem.cids(key, namespace="inchikey")
        return tuple(
            row.cid
            for row in await self.pubchem.compounds(ids[:4])
            if row.smiles == smiles
        )

    async def search(self, smiles: str, question: str) -> SourceSearch:
        """Find papers and patents for a structure and its analogs.

        Args:
            smiles: The structure to investigate.
            question: The target, assay, or other terms to match.

        Returns:
            Up to one matching paper and patent per structure scope, plus a
            topic paper if no exact paper matches. Failed lookups are listed separately.
        """
        smiles = canonicalize_smiles(smiles)
        unavailable: list[str] = []
        cids, exact, neighbors = await asyncio.gather(
            lookup("PubChem identity", self.exact_cids(smiles), unavailable),
            lookup("ChEMBL identity", self._exact_chembl(smiles), unavailable),
            self._neighbors(smiles, unavailable),
        )
        compounds = {
            cid: SourceLink("exact_pubchem", f"PUBCHEM/{cid}", smiles)
            for cid in cids or ()
        }
        for neighbor in neighbors:
            for cid in neighbor.cids:
                compounds.setdefault(
                    cid,
                    SourceLink(
                        "neighbor_pubchem",
                        f"PUBCHEM/{cid}",
                        neighbor.smiles,
                        neighbor.similarity,
                        neighbor.fingerprint_method,
                    ),
                )
        scope_cids = {
            scope: tuple(cid for cid, link in compounds.items() if link.scope == scope)
            for scope in STRUCTURE_SCOPES
        }
        aliases, matches, pubmed, patent_ids = await asyncio.gather(
            self._aliases(cids or (), unavailable),
            self._activity_matches(smiles, exact or (), neighbors, unavailable),
            self._cross_references(scope_cids, "PubMedID", unavailable),
            self._cross_references(scope_cids, "PatentID", unavailable),
        )
        query_terms = set(terms(question))
        activities = [match.activity for match in matches if match.activity is not None]
        criteria = _SourceQuery.parse(question, activities, aliases)
        paper_links, patent_links = await self._chembl_references(
            [
                match
                for match in matches
                if match.activity and criteria.matches_activity(match.activity)
            ],
            query_terms,
            unavailable,
        )
        for pmid, linked in _pubchem_references(
            pubmed or (), compounds, limit=MAX_DOCUMENTS
        ).items():
            if pmid.isascii() and pmid.isdigit():
                paper_links["EXT_ID", pmid].extend(linked)
        for ref, linked in _pubchem_references(
            patent_ids or (), compounds, limit=MAX_PATENTS
        ).items():
            patent_links[ref].extend(linked)
        papers, patents = await asyncio.gather(
            self._linked_papers(paper_links, unavailable),
            self._linked_patents(patent_links, unavailable),
        )

        passages = []
        seen: set[tuple[str, str]] = set()
        for scope in STRUCTURE_SCOPES:
            for source_type, documents in (("paper", papers), ("patent", patents)):
                selected = max(
                    (
                        (document, match)
                        for document, match in documents
                        if match.link.scope == scope
                        and (source_type, document.id) not in seen
                        and (match.activity or criteria.matches_document(document))
                    ),
                    key=lambda pair: _relevance(pair[1], query_terms, pair[0]),
                    default=None,
                )
                if selected:
                    document, match = selected
                    seen.add((source_type, document.id))
                    passages.append(
                        _passage(
                            source_type,
                            document,
                            match.link,
                            question,
                            aliases if scope == "exact" else (),
                        )
                    )
        if not any(
            p.source_type == "paper" and p.link.scope == "exact" for p in passages
        ):
            topic = await self._topic_paper(
                question, criteria, cids or (), aliases, unavailable
            )
            if topic and ("paper", topic[0].id) not in seen:
                passages.append(_passage("paper", *topic, question, aliases))
        return SourceSearch(tuple(passages), tuple(sorted(set(unavailable))))

    async def _neighbors(
        self, smiles: str, unavailable: list[str]
    ) -> tuple[_Neighbor, ...]:
        """Merge IDs for identical structures and rank the union once."""
        chembl, pubchem = await asyncio.gather(
            lookup(
                "ChEMBL neighbors",
                self.chembl.find_similar(smiles, threshold=80, max_records=50),
                unavailable,
            ),
            lookup(
                "PubChem neighbors",
                self.pubchem.find_similar(
                    smiles,
                    threshold=90,
                    max_records=50,
                    top_n=50,
                    single_component="." not in smiles,
                ),
                unavailable,
            ),
        )
        chembl_by_smiles: dict[str, list[ChEMBLMolecule]] = defaultdict(list)
        cids_by_smiles: dict[str, list[int]] = defaultdict(list)
        for molecule in chembl or ():
            if molecule.smiles:
                chembl_by_smiles[molecule.smiles].append(molecule)
        for hit in pubchem.hits if pubchem else ():
            if hit.compound.smiles:
                cids_by_smiles[hit.compound.smiles].append(hit.compound.cid)
        structures = sorted(
            structure
            for structure in (chembl_by_smiles.keys() | cids_by_smiles.keys())
            - {smiles}
            if "." in smiles or "." not in structure
        )
        if not structures:
            return ()
        fps = Fingerprints.from_mols([to_mol(text) for text in (smiles, *structures)])
        scores = DataStructs.BulkTanimotoSimilarity(fps.bits[0], fps.bits[1:])
        ranked = sorted(
            zip(structures, scores, strict=True), key=lambda item: (-item[1], item[0])
        )
        return tuple(
            _Neighbor(
                structure,
                tuple(chembl_by_smiles[structure]),
                tuple(dict.fromkeys(cids_by_smiles[structure])),
                score,
                fps.method,
            )
            for structure, score in ranked[:MAX_NEIGHBORS]
        )

    async def _aliases(
        self, cids: tuple[int, ...], unavailable: list[str]
    ) -> tuple[str, ...]:
        if not cids:
            return ()
        names = await lookup(
            "PubChem names", self.pubchem.synonyms(cids[:1], limit=3), unavailable
        )
        return tuple(
            name
            for name in (names or {}).get(cids[0], ())
            if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9 -]{1,40}", name)
        )

    async def _cross_references(
        self,
        scope_cids: Mapping[EvidenceScope, tuple[int, ...]],
        field: Literal["PatentID", "PubMedID"],
        unavailable: list[str],
    ) -> tuple[tuple[int, str], ...]:
        rows = await asyncio.gather(
            *(
                lookup(
                    f"PubChem {scope} {field}",
                    self.pubchem.cross_references(cids, field),
                    unavailable,
                )
                for scope, cids in scope_cids.items()
                if cids
            )
        )
        return tuple(item for batch in rows if batch for item in batch)

    async def _activity_matches(
        self,
        smiles: str,
        exact: tuple["ChEMBLMolecule", ...],
        neighbors: tuple[_Neighbor, ...],
        unavailable: list[str],
    ) -> list[_Match]:
        exact_ids = {molecule.chembl_id for molecule in exact}
        by_id = {
            molecule.chembl_id: neighbor
            for neighbor in neighbors
            for molecule in neighbor.chembl
            if molecule.chembl_id not in exact_ids
        }
        activities = await self._read_activities(
            (
                *exact,
                *(molecule for neighbor in neighbors for molecule in neighbor.chembl),
            ),
            unavailable,
        )
        matches = []
        for row in activities:
            ref = chembl_activity_ref(row.activity_id)
            neighbor = by_id.get(row.molecule_chembl_id)
            link = (
                SourceLink(
                    "neighbor_activity",
                    ref,
                    neighbor.smiles,
                    neighbor.similarity,
                    neighbor.fingerprint_method,
                )
                if neighbor is not None
                else SourceLink("exact_activity", ref, smiles)
            )
            matches.append(_Match(link, row))
        return matches

    async def _chembl_references(
        self,
        matches: list[_Match],
        query_terms: set[str],
        unavailable: list[str],
    ) -> tuple[dict[tuple[str, str], list[_Match]], dict[str, list[_Match]]]:
        selected: dict[str, list[_Match]] = defaultdict(list)
        ranked = sorted(
            matches, key=lambda match: _relevance(match, query_terms), reverse=True
        )
        for scope in STRUCTURE_SCOPES:
            refs: dict[str, _Match] = {}
            for match in ranked:
                if (
                    match.link.scope == scope
                    and match.activity
                    and match.activity.document_chembl_id
                ):
                    refs.setdefault(match.activity.document_chembl_id, match)
            for ref in list(refs)[:MAX_DOCUMENTS]:
                selected[ref].append(refs[ref])
        documents = await asyncio.gather(
            *(
                lookup(f"ChEMBL {ref}", self.chembl.document(ref), unavailable)
                for ref in selected
            )
        )
        papers: dict[tuple[str, str], list[_Match]] = defaultdict(list)
        patents: dict[str, list[_Match]] = defaultdict(list)
        for ref, doc in zip(selected, documents, strict=True):
            if not doc or doc.id != ref:
                continue
            if doc.patent_id:
                patents[doc.patent_id].extend(selected[ref])
            elif doc.doi and DOI.fullmatch(doc.doi):
                papers["DOI", doc.doi.casefold()].extend(selected[ref])
            elif doc.pmid and doc.pmid > 0:
                papers["EXT_ID", str(doc.pmid)].extend(selected[ref])
        return papers, patents

    async def _linked_papers(
        self, links: Mapping[tuple[str, str], list[_Match]], unavailable: list[str]
    ) -> list[tuple[Publication, _Match]]:
        if not links:
            return []
        query = " OR ".join(
            f"(EXT_ID:{value} AND SRC:MED)"
            if field == "EXT_ID"
            else f"DOI:{json.dumps(value)}"
            for field, value in links
        )
        rows = await lookup(
            "Europe PMC linked papers",
            self.papers.search(query, limit=max(12, 2 * len(links))),
            unavailable,
        )
        return [
            (article, match)
            for article in rows or ()
            for match in (
                *links.get(("DOI", article.doi), []),
                *links.get(("EXT_ID", article.pmid), []),
            )
        ]

    async def _linked_patents(
        self, links: Mapping[str, list[_Match]], unavailable: list[str]
    ) -> list[tuple[Publication, _Match]]:
        refs = []
        for scope in STRUCTURE_SCOPES:
            refs.extend(
                [
                    ref
                    for ref, matches in links.items()
                    if any(match.link.scope == scope for match in matches)
                ][:MAX_PATENTS]
            )
        chosen = tuple(dict.fromkeys(refs))
        documents = await asyncio.gather(
            *(
                lookup(ref, self.pubchem.patent_abstract(ref), unavailable)
                for ref in chosen
            )
        )
        return [
            (document, match)
            for ref, document in zip(chosen, documents, strict=True)
            if document is not None and document.id == ref
            for match in links[ref]
        ]

    async def _topic_paper(
        self,
        question: str,
        criteria: _SourceQuery,
        cids: tuple[int, ...],
        aliases: tuple[str, ...],
        unavailable: list[str],
    ) -> tuple[Publication, SourceLink] | None:
        topics = sorted(criteria.targets) if criteria.targets else terms(question)[:10]
        query = " OR ".join(f"TITLE_ABS:{json.dumps(topic)}" for topic in topics)
        if aliases:
            names = " OR ".join(f"TITLE_ABS:{json.dumps(name)}" for name in aliases)
            query = f"({names}) AND ({query})" if query else names
        if not query:
            return None
        rows = await lookup(
            "Europe PMC topic search", self.papers.search(query), unavailable
        )
        link = SourceLink("topic", f"PUBCHEM/{cids[0]}" if aliases else None)
        return select_document(
            [(row, link) for row in rows or () if criteria.matches_document(row)],
            question,
        )


def _passage(
    source_type: Literal["paper", "patent"],
    document: Publication,
    link: SourceLink,
    question: str,
    aliases: tuple[str, ...],
) -> Passage:
    idx, excerpt = select_excerpt(document.abstract, question, aliases)
    return Passage(
        passage_ref(source_type, document.id, idx),
        document.id,
        source_type,
        document.title,
        excerpt,
        document.url,
        link,
        pmcid=getattr(document, "pmcid", None),
        doi=getattr(document, "doi", None) or None,
        year=getattr(document, "year", None),
    )


def _missing_sources(found: SourceSearch, question: str) -> tuple[str, ...]:
    requested = [
        source_type
        for source_type, pattern in (
            ("paper", r"\b(?:papers?|publications?|articles?|literature)\b"),
            ("patent", r"\bpatent(?:s|ed|ing)?\b"),
        )
        if re.search(pattern, question, re.IGNORECASE)
    ]
    if not requested and found.passages:
        return ()
    present = {passage.source_type for passage in found.passages}
    missing = [
        f"{source_type}s"
        for source_type in requested or ("paper", "patent")
        if source_type not in present
    ]
    if not missing:
        return ()
    if found.unavailable:
        return (
            f"No matching {' or '.join(missing)} could be verified; some lookups were unavailable.",
        )
    return (f"No matching {' or '.join(missing)} were found in the sources checked.",)


def _merge_sources(found: SourceSearch, full_text: SourceSearch) -> SourceSearch:
    links_by_id: dict[str, SourceLink] = {}
    for passage in sorted(
        found.passages,
        key=lambda item: {"exact": 0, "analog": 1, "topic": 2}[item.link.scope],
    ):
        if passage.source_type == "paper":
            for ref in (passage.publication_id, passage.pmcid, passage.doi):
                if ref:
                    links_by_id.setdefault(ref, passage.link)
    passages = tuple(
        replace(
            passage,
            link=links_by_id.get(passage.pmcid)
            or links_by_id.get(passage.doi)
            or passage.link,
        )
        for passage in full_text.passages
    )
    return SourceSearch(
        (*passages, *found.passages),
        tuple(sorted(set((*found.unavailable, *full_text.unavailable)))),
        (*found.exclusions, *full_text.exclusions),
    )


class ResearchWorkflow:
    def __init__(self, sources: SourceFinder, task: ModelRunner) -> None:
        self.sources = sources
        self.task = task

    async def respond(
        self,
        *,
        molecule_id: MoleculeId,
        smiles: str,
        question: str,
        trace: ModelTrace | None = None,
        results: Sequence[ResultValue] = (),
    ) -> tuple[SourceSearch, Response]:
        """Find scientific evidence and synthesize cited findings.

        Retrieved passages remain available if synthesis fails.
        """
        trace = trace if trace is not None else self.task.new_trace()
        try:
            async with asyncio.timeout(RESEARCH_TIMEOUT_S):
                try:
                    plan = await plan_search(
                        self.task,
                        question,
                        molecule_id=molecule_id,
                        smiles=smiles,
                        trace=trace,
                    )
                except ProviderError:
                    plan = None
                focused = plan is not None and bool(plan.change_terms)
                if focused:
                    found, full_text = await asyncio.gather(
                        self.sources.search(smiles, question),
                        self.sources.find_passages(plan, question),
                    )
                    found = self.select_passages(
                        _merge_sources(found, full_text),
                        molecule_id,
                        smiles,
                        question,
                        results,
                    )
                else:
                    found = await self.sources.search(smiles, question)
                    if plan is None:
                        found = replace(
                            found,
                            unavailable=(
                                *found.unavailable,
                                "Full-text search planning",
                            ),
                        )
                limitations = _missing_sources(found, question)
                if focused and not any(
                    passage.license_url for passage in found.passages
                ):
                    limitations += (
                        "No CC BY full-text passages matched the requested change and property.",
                    )
                if plan is None:
                    limitations += (
                        "Full-text search planning is unavailable; the lookup used abstracts and patent excerpts.",
                    )
                if not found.passages and not results:
                    return found, Response((), limitations)
                try:
                    response = await respond(
                        self.task,
                        question,
                        molecules={molecule_id: smiles},
                        evidence=source_citations(found),
                        evidence_origin="external_document",
                        results=results,
                        trace=trace,
                    )
                except LimitError as exc:
                    return found, Response((), (*limitations, str(exc)))
                except ProviderError:
                    return (
                        replace(
                            found, unavailable=(*found.unavailable, "Source synthesis")
                        ),
                        Response(
                            (), (*limitations, "Source synthesis is unavailable.")
                        ),
                    )
                return found, Response(
                    response.findings, (*limitations, *response.limitations)
                )
        except TimeoutError as exc:
            raise ProviderError("The source search reached its time limit.") from exc

    def select_passages(
        self,
        found: SourceSearch,
        molecule_id: MoleculeId,
        smiles: str,
        question: str,
        results: Sequence[ResultValue],
    ) -> SourceSearch:
        """Fit whole passages within the model context, including retry feedback."""
        feedback = max(
            ValidationIssue, key=lambda issue: len(json.dumps(issue.value).encode())
        ).value
        selected = []
        exclusions = list(found.exclusions)
        for passage in found.passages:
            proposed = replace(found, passages=(*selected, passage))
            context = research_context(
                question,
                {molecule_id: smiles},
                source_citations(proposed),
                results,
                evidence_origin="external_document",
            )
            content = format_request(
                question, build_context(context), {"validation": feedback}
            )
            if len(content.encode()) <= self.task.runtime.max_context_bytes:
                selected.append(passage)
            else:
                exclusions.append(
                    SourceExclusion(
                        passage.publication_id,
                        "The passage exceeds the remaining synthesis context.",
                    )
                )
        return replace(
            found, passages=tuple(selected), exclusions=tuple(dict.fromkeys(exclusions))
        )

    async def activities(
        self, molecule_id: MoleculeId, smiles: str, question: str
    ) -> tuple[ResultValue, ...]:
        rows = await self.sources.exact_activities(smiles, question=question)
        query_terms = set(terms(question))
        ranked = sorted(
            rows,
            key=lambda row: (
                -len(
                    query_terms.intersection(
                        terms(f"{row.target_name} {row.assay_description}")
                    )
                )
            ),
        )
        return tuple(
            ResultValue(
                id=chembl_activity_ref(row.activity_id),
                molecule_id=molecule_id,
                endpoint=row.standard_type,
                value=row.standard_value,
                unit=row.standard_units,
                method="ChEMBL standardized activity",
                source_type="experimental",
                relation=row.standard_relation,
                assay_id=row.assay_chembl_id,
                target=row.target_name,
                assay_description=row.assay_description,
            )
            for row in ranked
            if row.standard_type
            and row.standard_units
            and row.standard_relation
            and row.standard_value is not None
            and isfinite(row.standard_value)
        )[:12]


def source_citations(found: SourceSearch) -> tuple[Citation, ...]:
    evidence = []
    relations = {
        "exact_activity": "Linked by an activity record for the exact structure.",
        "exact_pubchem": "Indexed under the exact PubChem structure.",
        "neighbor_activity": "Linked by a ChEMBL activity record for a structural analog.",
        "neighbor_pubchem": "Indexed under a structural analog in PubChem.",
        "topic": "Text match. Exact structure attribution is unverified.",
    }
    for passage in found.passages:
        scope = relations[passage.link.relation]
        if passage.source_type == "patent":
            scope += " Patent claim scope is unreviewed."
        structure = (
            f"Matched analog SMILES: {passage.link.matched_smiles}\n"
            if passage.link.scope == "analog"
            else ""
        )
        provenance = "Abstract excerpt."
        if passage.license_url:
            provenance = (
                f"Full-text quote. License: {passage.license_url}\n"
                f"PMCID: {passage.pmcid}. DOI: {passage.doi or 'unavailable'}. "
                f"Location: {passage.locator}."
            )
        evidence.append(
            Citation(
                passage.id,
                f"{scope}\n{structure}{provenance}\n{passage.title}\n{passage.text}",
                tuple(
                    dict.fromkeys(
                        ref
                        for ref in (
                            passage.publication_id,
                            passage.publication_id.replace("-", ""),
                            passage.pmcid,
                            passage.doi,
                        )
                        if ref
                    )
                ),
            )
        )
    return tuple(evidence)
