import logging
import re
from collections.abc import Iterator
from xml.etree import ElementTree

import httpx

from henet_kb.ingest.base import Document
from henet_kb.ingest.clean import html_to_document
from henet_kb.ingest.http import make_client

log = logging.getLogger(__name__)

SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

# Yoast publishes one sitemap per taxonomy and per JetPlugins menu. Those are
# navigation pages, not content, so they are skipped by default.
DEFAULT_EXCLUDE = (
    r"/(jet-menu|jet-popup|category|post_tag|tax_cidade)-sitemap\.xml$",
    r"/(category|tag|cidade|author)/",
    r"\.(pdf|jpe?g|png|gif|webp|svg|mp4|zip)$",
    r"/cdn-cgi/",
    r"/wp-content/",
)


class SitemapCrawler:
    """Reads every page listed in the site's XML sitemaps."""

    name = "sitemap"

    def __init__(
        self,
        site_url: str,
        sitemap_path: str = "/sitemap_index.xml",
        exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
        client: httpx.Client | None = None,
        max_pages: int | None = None,
    ) -> None:
        self.site_url = site_url.rstrip("/")
        self.sitemap_url = self.site_url + sitemap_path
        self.exclude = [re.compile(pattern, re.IGNORECASE) for pattern in exclude]
        self.client = client or make_client()
        self.max_pages = max_pages

    def _excluded(self, url: str) -> bool:
        return any(pattern.search(url) for pattern in self.exclude)

    def _fetch_locs(self, url: str) -> tuple[list[str], bool]:
        response = self.client.get(url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        is_index = root.tag == f"{SITEMAP_NS}sitemapindex"
        locs = [
            element.text.strip()
            for element in root.iter(f"{SITEMAP_NS}loc")
            if element.text and element.text.strip()
        ]
        return locs, is_index

    def urls(self) -> list[str]:
        seen: dict[str, None] = {}
        pending = [self.sitemap_url]
        while pending:
            sitemap = pending.pop(0)
            if self._excluded(sitemap):
                continue
            try:
                locs, is_index = self._fetch_locs(sitemap)
            except (httpx.HTTPError, ElementTree.ParseError) as exc:
                log.warning("skipping sitemap %s: %s", sitemap, exc)
                continue
            if is_index:
                pending.extend(locs)
                continue
            for loc in locs:
                normalized = loc.rstrip("/") or loc
                if not self._excluded(normalized):
                    seen.setdefault(normalized, None)
        return list(seen)

    def documents(self) -> Iterator[Document]:
        for count, url in enumerate(self.urls()):
            if self.max_pages is not None and count >= self.max_pages:
                return
            try:
                response = self.client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("skipping page %s: %s", url, exc)
                continue
            if "html" not in response.headers.get("content-type", ""):
                continue
            document = html_to_document(response.text, url=url, source=self.name)
            if document.text:
                yield document
