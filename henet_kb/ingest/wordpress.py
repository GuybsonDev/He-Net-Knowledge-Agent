import html
import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import httpx

from henet_kb.ingest.base import Document
from henet_kb.ingest.clean import html_to_document
from henet_kb.ingest.http import make_client

log = logging.getLogger(__name__)

LEADING_MARKUP_RE = re.compile(r"^\s*(?:<style[^>]*>.*?</style>\s*|<[^>]+>\s*)+", re.DOTALL)


def parse_json_body(text: str) -> Any:
    """Parse a REST body that may have stray markup in front of the JSON.

    Elementor prints inline style tags while rendering content, and they end up
    before the JSON when _fields asks for content.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        stripped = LEADING_MARKUP_RE.sub("", text)
        if not stripped or stripped[0] not in "[{":
            raise
        return json.loads(stripped)


class WordPressRestSource:
    """Reads pages and posts through the WordPress REST API (wp/v2)."""

    name = "wordpress"

    def __init__(
        self,
        site_url: str,
        post_types: tuple[str, ...] = ("pages", "posts"),
        per_page: int = 100,
        client: httpx.Client | None = None,
    ) -> None:
        self.site_url = site_url.rstrip("/")
        self.post_types = post_types
        self.per_page = per_page
        self.client = client or make_client()

    def _items(self, post_type: str) -> Iterator[dict]:
        page = 1
        while True:
            response = self.client.get(
                f"{self.site_url}/wp-json/wp/v2/{post_type}",
                params={
                    "per_page": self.per_page,
                    "page": page,
                    "status": "publish",
                    "_fields": "id,link,title,content,modified",
                },
            )
            if response.status_code == 400 and page > 1:
                # WordPress answers 400 once you page past the end.
                return
            response.raise_for_status()
            items = parse_json_body(response.text)
            if not items:
                return
            yield from items
            total_pages = int(response.headers.get("X-WP-TotalPages", "1"))
            if page >= total_pages:
                return
            page += 1

    def documents(self) -> Iterator[Document]:
        for post_type in self.post_types:
            try:
                items = list(self._items(post_type))
            except httpx.HTTPError as exc:
                log.warning("skipping %s: %s", post_type, exc)
                continue
            for item in items:
                title = html.unescape(item.get("title", {}).get("rendered", "")).strip()
                content = item.get("content", {}).get("rendered", "")
                document = html_to_document(
                    content,
                    url=item["link"].rstrip("/") or item["link"],
                    source=self.name,
                    title=title or None,
                    modified=item.get("modified"),
                )
                if document.text:
                    yield document
