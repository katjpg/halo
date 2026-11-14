import json
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from html import unescape
from urllib.parse import quote, urlsplit
from xml.etree.ElementTree import Element, TreeBuilder
from xml.parsers.expat import ExpatError, ParserCreate

from halo.clients.http import (
    HttpClient,
    ResponseError,
    ServiceClient,
    check_integer,
    json_object,
    records,
    text,
)
from halo.config.settings import DEFAULT_INTEGRATIONS
from halo.core.errors import ParseError
from halo.models.research import Article, FullText, Paragraph

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
DOI = re.compile(r"10\.\d{4,9}/[-._;()/a-z0-9]+", re.IGNORECASE)
SEARCH_LIMIT = 1000


class EuropePMCClient(ServiceClient):
    def __init__(
        self,
        http: HttpClient | None = None,
        *,
        requests_per_second: float = DEFAULT_INTEGRATIONS.europe_pmc_requests_per_second,
    ) -> None:
        super().__init__(BASE_URL, http, requests_per_second=requests_per_second)

    async def article(self, identifier: str) -> Article | None:
        """Find metadata by DOI, PMID, or PMCID, including records without abstracts."""
        field, value = _article_identifier(identifier)
        query_field = "EXT_ID" if field == "pmid" else field.upper()
        query_value = json.dumps(value) if field == "doi" else value
        query = f"{query_field}:{query_value}"
        if field == "pmid":
            query += " AND SRC:MED"
        rows = await self._search_rows(query, limit=2)
        for row in rows:
            article = _parse_article(row, require_abstract=False)
            if (
                article is not None
                and row.get("source") != "PAT"
                and getattr(article, field) == value
            ):
                return article
        if rows:
            raise ResponseError(
                "Europe PMC returned invalid or mismatched article metadata."
            )
        return None

    async def search(self, query: str, *, limit: int = 12) -> tuple[Article, ...]:
        """Return non-patent records that have an abstract and a usable ID."""
        return await self._search(query, limit=limit, require_abstract=True)

    async def search_full_text(
        self, query: str, *, limit: int = 5
    ) -> tuple[Article, ...]:
        """Find open-access candidates indexed as CC BY; verify their XML before use."""
        if not query.strip():
            raise ParseError("query cannot be empty")
        check_integer(limit, "limit", maximum=5)
        return await self._search(
            f'({query}) AND OPEN_ACCESS:Y AND LICENSE:"CC BY"',
            limit=limit,
            require_abstract=False,
        )

    async def full_text(self, pmcid: str) -> FullText:
        """Read an article's main text and license from its JATS XML."""
        if not re.fullmatch(r"PMC[1-9]\d*", pmcid):
            raise ParseError("Provide a PMCID such as PMC123456.")
        content = await self._http.request_xml(
            "GET", (pmcid, "fullTextXML"), cache=True
        )
        return _parse_full_text(content, pmcid)

    async def _search(
        self, query: str, *, limit: int, require_abstract: bool
    ) -> tuple[Article, ...]:
        rows = await self._search_rows(query, limit=limit)
        return tuple(
            item
            for row in rows
            if row.get("source") != "PAT"
            and (item := _parse_article(row, require_abstract=require_abstract))
        )

    async def _search_rows(self, query: str, *, limit: int) -> list[dict[str, object]]:
        query = query.strip()
        if not query:
            raise ParseError("query cannot be empty")
        check_integer(limit, "limit", maximum=SEARCH_LIMIT)
        payload = json_object(
            await self._http.request_json(
                "GET",
                "search",
                params={
                    "query": query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": limit,
                },
                cache=True,
            ),
            "Europe PMC search result",
        )
        return records(
            json_object(payload.get("resultList"), "Europe PMC result list"), "result"
        )


def _article_identifier(identifier: str) -> tuple[str, str]:
    if not isinstance(identifier, str):
        raise ParseError("Provide a DOI, PMID, or PMCID.")
    value = identifier.strip()
    prefix = None
    if value.casefold().startswith(("https://doi.org/", "http://doi.org/")):
        prefix, value = "doi", value.split("/", 3)[-1]
    elif match := re.match(r"^(doi|pmid|pmcid):\s*(.*)$", value, re.IGNORECASE):
        prefix, value = match[1].casefold(), match[2]
    if prefix in (None, "pmcid") and re.fullmatch(r"PMC[1-9]\d*", value, re.IGNORECASE):
        return "pmcid", value.upper()
    if prefix in (None, "pmid") and re.fullmatch(r"[1-9]\d*", value):
        return "pmid", value
    if prefix in (None, "doi") and DOI.fullmatch(value):
        return "doi", value.casefold()
    raise ParseError("Provide a DOI, PMID, or PMCID.")


def _parse_article(
    row: Mapping[str, object], *, require_abstract: bool = True
) -> Article | None:
    title = _plain_text(text(row, "title") or "")
    abstract = _plain_text(text(row, "abstractText") or "")
    if not title or (require_abstract and not abstract):
        return None
    pmcid, pmid, doi = (text(row, field) or "" for field in ("pmcid", "pmid", "doi"))
    if re.fullmatch(r"PMC\d+", pmcid):
        ref, url = pmcid, f"https://europepmc.org/articles/{pmcid}"
    elif pmid.isdigit():
        ref, url = f"PMID-{pmid}", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    elif DOI.fullmatch(doi):
        ref, url = doi.casefold(), f"https://doi.org/{quote(doi, safe='/')}"
    else:
        return None
    year = text(row, "pubYear") or ""
    return Article(
        id=ref,
        title=title,
        abstract=abstract,
        url=url,
        doi=doi.casefold(),
        pmid=pmid,
        pmcid=pmcid if re.fullmatch(r"PMC[1-9]\d*", pmcid) else None,
        year=int(year) if re.fullmatch(r"\d{4}", year) else None,
    )


def _expanded_name(name: str) -> str:
    return f"{{{name}" if "}" in name else name


def _parse_xml(content: bytes) -> Element:
    builder = TreeBuilder()
    parser = ParserCreate(namespace_separator="}")
    parser.StartElementHandler = lambda name, attrs: builder.start(
        _expanded_name(name),
        {_expanded_name(key): value for key, value in attrs.items()},
    )
    parser.EndElementHandler = lambda name: builder.end(_expanded_name(name))
    parser.CharacterDataHandler = builder.data

    def reject_entity(*args: object) -> None:
        raise ResponseError("Article XML must not declare or load entities.")

    parser.EntityDeclHandler = reject_entity
    parser.ExternalEntityRefHandler = reject_entity
    try:
        parser.Parse(content, True)
    except ExpatError as exc:
        raise ResponseError("Europe PMC returned invalid article XML.") from exc
    root = builder.close()
    if root.tag.rsplit("}", 1)[-1] != "article":
        raise ResponseError("Europe PMC returned no JATS article.")
    return root


def _parse_full_text(content: bytes, pmcid: str) -> FullText:
    root = _parse_xml(content)
    metadata = root.find("./{*}front/{*}article-meta")
    if metadata is None:
        raise ResponseError("Article XML has no article metadata.")
    identifiers = {
        item.get("pub-id-type"): "".join(item.itertext()).strip()
        for item in metadata.findall("./{*}article-id")
    }
    stored_pmcid = identifiers.get("pmc") or identifiers.get("pmcid")
    if stored_pmcid and stored_pmcid.removeprefix("PMC") != pmcid[3:]:
        raise ResponseError("Article XML belongs to a different PMCID.")
    doi = identifiers.get("doi", "").casefold()
    title = metadata.find("./{*}title-group/{*}article-title")
    year = metadata.findtext("./{*}pub-date/{*}year", "").strip()
    return FullText(
        pmcid=pmcid,
        title=" ".join("".join(title.itertext()).split()) if title is not None else "",
        doi=doi if DOI.fullmatch(doi) else None,
        year=int(year) if re.fullmatch(r"\d{4}", year) else None,
        license_url=_read_license(metadata),
        paragraphs=_read_paragraphs(root.find("./{*}body")),
    )


def _read_license(metadata: Element) -> str | None:
    license_urls = set()
    today = datetime.now(UTC).date()
    for license_element in metadata.findall("./{*}permissions/{*}license"):
        for item in license_element.iter():
            if start := item.get("start_date"):
                try:
                    effective_date = date.fromisoformat(start)
                except ValueError:
                    continue
                if effective_date > today:
                    continue
            value = item.get("{http://www.w3.org/1999/xlink}href")
            if item.tag.rsplit("}", 1)[-1] == "license_ref":
                value = "".join(item.itertext()).strip()
            if value and (url := _cc_license_url(value)):
                license_urls.add(url)
    return next(iter(license_urls)) if len(license_urls) == 1 else None


def _read_paragraphs(body: Element | None) -> tuple[Paragraph, ...]:
    if body is None:
        return ()
    excluded = {
        "fig",
        "fig-group",
        "table",
        "table-wrap",
        "table-wrap-group",
        "supplementary-material",
        "ref-list",
        "ref",
        "fn-group",
    }
    for parent in body.iter():
        for i, child in reversed(list(enumerate(parent))):
            if child.tag.rsplit("}", 1)[-1] in excluded:
                if child.tail:
                    if i:
                        parent[i - 1].tail = (parent[i - 1].tail or "") + child.tail
                    else:
                        parent.text = (parent.text or "") + child.tail
                parent.remove(child)
    paragraphs = []
    for paragraph in body.iterfind(".//{*}p"):
        value = "".join(paragraph.itertext()).strip()
        if value:
            paragraphs.append(Paragraph(f"body/paragraph-{len(paragraphs) + 1}", value))
    return tuple(paragraphs)


def _cc_license_url(value: str) -> str | None:
    try:
        url = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        url.scheme not in {"http", "https"}
        or url.netloc.casefold()
        not in {"creativecommons.org", "www.creativecommons.org"}
        or url.query
        or url.fragment
    ):
        return None
    match = re.fullmatch(
        r"/licenses/(by(?:-nc)?(?:-nd|-sa)?)/(1\.0|2\.0|2\.5|3\.0|4\.0)/?"
        r"(?:legalcode|deed(?:\.[a-z-]+)?)?/?",
        url.path,
    )
    if match is None:
        return None
    return f"https://creativecommons.org/licenses/{match[1]}/{match[2]}/"


def _plain_text(value: str) -> str:
    # Strip tags before unescaping so escaped markup stays literal.
    return " ".join(unescape(re.sub(r"</?[A-Za-z][^>]*>", " ", value)).split())
