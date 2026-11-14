from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from halo.clients.http import (
    HttpClient,
    ResponseError,
    ServiceClient,
    check_integer,
    integer,
    json_object,
    number,
    records,
    text,
)
from halo.config.settings import DEFAULT_INTEGRATIONS
from halo.core.errors import ParseError

BASE_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,corpusId,externalIds,title,abstract,year,venue,authors,url"
SEARCH_LIMIT = 100
SEARCH_WINDOW = 1000
SNIPPET_LIMIT = 1000
BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class Author:
    author_id: str | None
    name: str | None


@dataclass(frozen=True, slots=True)
class Paper:
    """A paper with external IDs keyed by the service's identifier names."""

    paper_id: str
    corpus_id: int | None
    external_ids: Mapping[str, str]
    title: str | None
    abstract: str | None
    year: int | None
    venue: str | None
    authors: tuple[Author, ...]
    url: str | None


@dataclass(frozen=True)
class PaperSearch:
    """One page of relevance-ranked search results.

    Total matches can exceed the service's paging window.
    """

    query: str
    total: int
    papers: tuple[Paper, ...]


@dataclass(frozen=True, slots=True)
class Snippet:
    """A paper passage with its type and any section heading."""

    text: str
    snippet_type: str | None
    section: str | None
    score: float | None
    paper_id: str | None
    corpus_id: int | None
    title: str | None


@dataclass(frozen=True)
class PaperLookup:
    """Papers found by identifier, and the identifiers the service did not know."""

    papers: Mapping[str, Paper]
    missing: tuple[str, ...]


class SemanticScholarClient(ServiceClient):
    def __init__(
        self,
        http: HttpClient | None = None,
        *,
        api_key: str | None = None,
        requests_per_second: float = DEFAULT_INTEGRATIONS.semantic_scholar_requests_per_second,
    ) -> None:
        super().__init__(
            BASE_URL,
            http,
            requests_per_second=requests_per_second,
            headers={"x-api-key": api_key} if api_key else None,
        )

    async def search_papers(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        fields_of_study: Sequence[str] = (),
        year: str | None = None,
    ) -> PaperSearch:
        """Return one page of relevance-ranked papers.

        Normalize hyphens and whitespace to spaces. Limit is 1 to 100 and
        offset + limit is at most 1000. Year accepts a year or a range with
        either end optional.
        """
        terms = _normalize_query(query)
        check_integer(limit, "limit", maximum=SEARCH_LIMIT)
        check_integer(offset, "offset", minimum=0, maximum=SEARCH_WINDOW - limit)
        if isinstance(fields_of_study, str):
            raise ParseError("fields_of_study must be a sequence of names")
        params: dict[str, str | int] = {
            "query": terms,
            "fields": PAPER_FIELDS,
            "limit": limit,
            "offset": offset,
        }
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(dict.fromkeys(fields_of_study))
        if year:
            params["year"] = year
        payload = json_object(
            await self._http.request_json(
                "GET", "paper/search", params=params, cache=True
            ),
            "a paper search result",
        )
        total = integer(payload, "total")
        if total is None or total < 0:
            raise ResponseError("paper search is missing its result count")
        rows = records(payload, "data", default=[])
        return PaperSearch(terms, total, tuple(_parse_paper(row) for row in rows))

    async def search_snippets(
        self, query: str, *, limit: int = 10
    ) -> tuple[Snippet, ...]:
        """Return passages in relevance order, with a limit from 1 to 1000."""
        terms = _normalize_query(query)
        check_integer(limit, "limit", maximum=SNIPPET_LIMIT)
        payload = json_object(
            await self._http.request_json(
                "GET",
                "snippet/search",
                params={"query": terms, "limit": limit},
                cache=True,
            ),
            "a snippet search result",
        )
        rows = records(payload, "data", default=[])
        return tuple(_parse_snippet(row) for row in rows)

    async def papers(self, ids: Sequence[str]) -> PaperLookup:
        """Look up papers by paper ID or a prefixed ID such as DOI or CorpusId."""
        if isinstance(ids, str) or any(
            not isinstance(name, str) or not name.strip() for name in ids
        ):
            raise ParseError("give a sequence of nonempty identifiers")
        requested_ids = list(dict.fromkeys(name.strip() for name in ids))
        found: dict[str, Paper] = {}
        missing: list[str] = []
        for start in range(0, len(requested_ids), BATCH_SIZE):
            batch = requested_ids[start : start + BATCH_SIZE]
            payload = await self._http.request_json(
                "POST",
                "paper/batch",
                params={"fields": PAPER_FIELDS},
                json_body={"ids": batch},
                replayable=True,
                cache=True,
            )
            if not isinstance(payload, list) or len(payload) != len(batch):
                raise ResponseError(
                    "paper batch does not match the identifiers asked for"
                )
            for name, row in zip(batch, payload, strict=True):
                if row is None:
                    missing.append(name)
                else:
                    found[name] = _parse_paper(json_object(row, "a paper record"))
        return PaperLookup(found, tuple(missing))


def _normalize_query(query: str) -> str:
    terms = " ".join(query.replace("-", " ").split())
    if not terms:
        raise ParseError("query cannot be empty")
    return terms


def _parse_paper(row: Mapping[str, Any]) -> Paper:
    paper_id = text(row, "paperId")
    if not paper_id:
        raise ResponseError("paper record is missing paperId")
    return Paper(
        paper_id=paper_id,
        corpus_id=integer(row, "corpusId"),
        external_ids=_external_ids(row.get("externalIds")),
        title=text(row, "title"),
        abstract=text(row, "abstract"),
        year=integer(row, "year"),
        venue=text(row, "venue"),
        authors=_authors(row.get("authors")),
        url=text(row, "url"),
    )


def _external_ids(value: object) -> Mapping[str, str]:
    if value is None:
        return {}
    record = json_object(value, "externalIds")
    return {key: str(val) for key, val in record.items() if val is not None}


def _authors(value: object) -> tuple[Author, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ResponseError("expected a list of authors")
    return tuple(
        Author(text(row, "authorId"), text(row, "name"))
        for row in (json_object(item, "an author record") for item in value)
    )


def _parse_snippet(row: Mapping[str, Any]) -> Snippet:
    passage = json_object(row.get("snippet"), "a snippet")
    body = text(passage, "text")
    if body is None:
        raise ResponseError("snippet is missing its text")
    paper = row.get("paper")
    paper = {} if paper is None else json_object(paper, "a snippet's paper")
    return Snippet(
        text=body,
        snippet_type=text(passage, "snippetKind"),
        section=text(passage, "section"),
        score=number(row, "score"),
        paper_id=text(paper, "paperId"),
        corpus_id=integer(paper, "corpusId"),
        title=text(paper, "title"),
    )
