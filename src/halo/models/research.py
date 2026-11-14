import re
from dataclasses import dataclass
from typing import Literal

from halo.types import MoleculeId

SourceRelation = Literal[
    "exact_activity", "exact_pubchem", "neighbor_activity", "neighbor_pubchem", "topic"
]
EvidenceScope = Literal["exact", "analog", "topic"]
PATENT_NUMBER = re.compile(r"([A-Z]{2})-?(\d+)-?([A-Z]\d?)")


def publication_number(value: str) -> str | None:
    """Normalize a patent publication number for a PubChem lookup."""
    match = PATENT_NUMBER.fullmatch(value.strip().upper())
    return "-".join(match.groups()) if match else None


@dataclass(frozen=True)
class Publication:
    id: str
    title: str
    abstract: str
    url: str


@dataclass(frozen=True)
class Article(Publication):
    doi: str
    pmid: str
    pmcid: str | None = None
    year: int | None = None


@dataclass(frozen=True)
class Paragraph:
    locator: str
    text: str


@dataclass(frozen=True)
class FullText:
    pmcid: str
    title: str
    doi: str | None
    year: int | None
    license_url: str | None
    paragraphs: tuple[Paragraph, ...]


@dataclass(frozen=True)
class SourceLink:
    """Link a publication to a structure and its optional Morgan Tanimoto score."""

    relation: SourceRelation
    source_ref: str | None = None
    matched_smiles: str | None = None
    morgan_tanimoto: float | None = None
    fingerprint_method: str | None = None

    def __post_init__(self) -> None:
        if self.relation == "topic":
            if self.matched_smiles is not None:
                raise ValueError("A text match has no verified structure.")
        elif not self.source_ref or not self.matched_smiles:
            raise ValueError("A structure link needs its source ID and SMILES.")
        if self.morgan_tanimoto is not None:
            if self.scope != "analog" or not 0 <= self.morgan_tanimoto <= 1:
                raise ValueError(
                    "A Morgan Tanimoto score needs an analog and lies in [0, 1]."
                )
            if not self.fingerprint_method:
                raise ValueError("A similarity score needs a fingerprint method.")
        elif self.fingerprint_method is not None:
            raise ValueError("A fingerprint method needs a similarity score.")

    @property
    def scope(self) -> EvidenceScope:
        if self.relation.startswith("exact_"):
            return "exact"
        if self.relation.startswith("neighbor_"):
            return "analog"
        return "topic"


@dataclass(frozen=True)
class Passage:
    id: str
    publication_id: str
    source_type: Literal["paper", "patent"]
    title: str
    text: str
    url: str
    link: SourceLink
    pmcid: str | None = None
    doi: str | None = None
    year: int | None = None
    license_url: str | None = None
    locator: str | None = None


@dataclass(frozen=True)
class SourceExclusion:
    publication_id: str
    reason: str


@dataclass(frozen=True)
class SourceSearch:
    passages: tuple[Passage, ...]
    unavailable: tuple[str, ...] = ()
    exclusions: tuple[SourceExclusion, ...] = ()

    def for_scope(self, scope: EvidenceScope) -> tuple[Passage, ...]:
        """Return passages with the requested structure relationship."""
        return tuple(
            passage for passage in self.passages if passage.link.scope == scope
        )


@dataclass(frozen=True)
class Citation:
    id: str
    text: str
    identifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finding:
    text: str
    molecule_ids: tuple[MoleculeId, ...]
    evidence_ids: tuple[str, ...]
    result_ids: tuple[str, ...]


@dataclass(frozen=True)
class Response:
    findings: tuple[Finding, ...]
    limitations: tuple[str, ...]
