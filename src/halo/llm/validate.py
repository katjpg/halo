import re
from collections.abc import Iterable
from enum import StrEnum
from unicodedata import normalize

from halo.core.errors import ProviderError


class ValidationIssue(StrEnum):
    """Application messages that may be sent as retry feedback."""

    OUTSIDE_CONTEXT = "The response cites an item outside the selected context."
    RESULT_MOLECULE = "A cited result needs its molecule ID in the finding."
    ASSAY_COVERAGE = "Cite a result for each molecule in the assay finding."
    INCOMPARABLE_ASSAYS = (
        "Report qualified values and results from different assays or conditions "
        "in separate findings."
    )
    MISSING_REFERENCE = "Each finding needs a reference."
    MISSING_LIMITATION = "A response without findings needs a limitation."
    MISSING_TEXT = "The response needs text."
    UNVERIFIED_QUANTITY = (
        "Remove numerical values and measurement phrases from prose; cite their "
        "structured results or source quotes. Preserve chemical locants and cited labels."
    )
    INVALID_CANDIDATES = (
        "Choose distinct eligible candidates and explain the selection."
    )
    UNKNOWN_DECISION = "The rationale cites an unknown decision."
    INVALID_OUTPUT = "Check the supplied references, eligible IDs, and text rules."
    UNVERIFIED_ROUTE = "Copy assay and endpoint labels from the request."
    INVALID_STRUCTURE = (
        "Preserve the source scaffold, stereochemistry, and any mapped attachment site."
    )
    DUPLICATE_STRUCTURE = "A proposal repeats a known structure."
    MISSING_PROPOSALS = "Return at least one candidate."
    PROPOSAL_LIMIT = "Return up to the requested number of candidates."


class OutputValidationError(ProviderError):
    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.value)
        self.issue = issue


NUMBER = re.compile(r"\d")
NAME_TOKEN = re.compile(r"\w+(?:[.,:/()+-]+\w+)*\)?")
QUANTITY = re.compile(
    r"[+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?(?:[a-zµμ]+(?:/[a-zµμ]+)?|-[a-zµμ]{1,4})?%?\)?",
    re.IGNORECASE,
)
MEASUREMENT_UNIT = (
    r"\s*%|(?:\s*|-)(?:[fpnuµμmk](?:m|mol|g|l)|mol|fold|percent)\b|"
    r"(?:\s+|-)(?:[mgl]|(?:femto|pico|nano|micro|milli|kilo)?"
    r"(?:molar|moles?|grams?|liters?|litres?))\b"
)
NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|half|quarter)"
)
MEASUREMENT = re.compile(
    rf"(?<![\w./])(?:\d+(?:[.,]\d+)?(?:e[+-]?\d+)?|"
    rf"{NUMBER_WORD}(?:[ -](?:and )?{NUMBER_WORD})*)(?:{MEASUREMENT_UNIT})"
)
LABEL_ID = rf"\d+[a-z]?(?!\w|[./-]\d|{MEASUREMENT_UNIT})"
CITED_LABEL = re.compile(
    rf"\b(?P<label>claims?|compounds?|examples?|figures?|tables?)\s+"
    rf"(?P<ids>{LABEL_ID}(?:\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+){LABEL_ID})*)"
)
LOCANT = r"(?:[1-9]\d?[a-z]?|[nops])['′]?"
LOCANTS = rf"{LOCANT}(?:,{LOCANT})*"
CHEMICAL_STEM = (
    r"(?:mono|di|tri|tetra|penta|hexa)?"
    r"(?:acet|alk|amin|anilin|azetid|benz|brom|but|carbox|chlor|cyclo|cyan|eth|"
    r"fluor|fur|hex|hydrox|imidaz|indol|iod|isoxaz|meth|morpholin|naphth|nitro|"
    r"oxaz|oxo|pent|phen|piper|prop|pyraz|pyrid|pyrimid|sulfon|thiaz|thien|triaz)"
)
LEADING_LOCANTS = re.compile(
    rf"(?<![\w/.,+])(?:{LOCANTS}|\({LOCANTS}\))-(?=\(?{CHEMICAL_STEM}[a-z]*\b)"
)
SUFFIX_LOCANTS = re.compile(
    rf"(?<![\w/]){CHEMICAL_STEM}[a-z]*-{LOCANTS}-(?:yl|diyl|ol|diol|one|dione|amine)\b"
)
DASHES = str.maketrans("‐‑–−", "----")


def normalize_text(value: str) -> str:
    return normalize("NFKC", value).casefold().translate(DASHES)


def cited_names(value: str) -> set[str]:
    value = normalize_text(value)
    tokens = set(NAME_TOKEN.findall(value))
    tokens.update(NAME_TOKEN.findall(value.replace("(", " ").replace(")", " ")))
    return {
        token
        for token in tokens
        if NUMBER.search(token)
        and not QUANTITY.fullmatch(token)
        and any(char.isalpha() for char in token)
        and ("-" in token or any(char.isdigit() for char in token))
    }


def cited_labels(value: str) -> set[str]:
    """Read numbered labels without allowing their numbers on their own."""
    labels = set()
    for match in CITED_LABEL.finditer(normalize_text(value)):
        label = match["label"].removesuffix("s")
        labels.add(" ".join(match[0].split()))
        labels.update(f"{label} {ref}" for ref in re.findall(LABEL_ID, match["ids"]))
    return labels


def validate_text(text: str, *, names: Iterable[str]) -> None:
    """Allow chemical locants and cited labels while keeping measurements structured."""
    if not text.strip():
        raise OutputValidationError(ValidationIssue.MISSING_TEXT)
    remaining = " ".join(normalize_text(text).split())
    if MEASUREMENT.search(remaining):
        raise OutputValidationError(ValidationIssue.UNVERIFIED_QUANTITY)
    allowed = sorted(
        {
            normalize_text(name)
            for name in names
            if name and not QUANTITY.fullmatch(normalize_text(name))
        },
        key=len,
        reverse=True,
    )
    if allowed:
        pattern = r"(?<![\w/])(?:" + "|".join(map(re.escape, allowed)) + r")(?![\w/])"
        remaining = re.sub(pattern, "", remaining)
    remaining = SUFFIX_LOCANTS.sub("", LEADING_LOCANTS.sub("", remaining))
    if NUMBER.search(remaining):
        raise OutputValidationError(ValidationIssue.UNVERIFIED_QUANTITY)
