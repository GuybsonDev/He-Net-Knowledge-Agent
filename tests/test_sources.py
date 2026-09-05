import httpx
import pytest
import respx

from henet_kb.ingest.sitemap import SitemapCrawler
from henet_kb.ingest.wordpress import WordPressRestSource

SITE = "https://example.test"

INDEX = f"""<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>{SITE}/page-sitemap.xml</loc></sitemap>
  <sitemap><loc>{SITE}/category-sitemap.xml</loc></sitemap>
</sitemapindex>"""

PAGES = f"""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{SITE}/</loc></url>
  <url><loc>{SITE}/internet/</loc></url>
  <url><loc>{SITE}/wp-content/uploads/contrato.pdf</loc></url>
</urlset>"""

PAGE_HTML = """<html><head><title>Internet | Exemplo</title></head>
<body><main><h2>Planos</h2><p>Fibra de 500 mega.</p></main></body></html>"""


@pytest.fixture
def client():
    return httpx.Client(follow_redirects=True)


@respx.mock
def test_sitemap_crawler_follows_index_and_skips_excluded(client):
    respx.get(f"{SITE}/sitemap_index.xml").mock(return_value=httpx.Response(200, text=INDEX))
    respx.get(f"{SITE}/page-sitemap.xml").mock(return_value=httpx.Response(200, text=PAGES))
    category = respx.get(f"{SITE}/category-sitemap.xml")
    respx.get(f"{SITE}/").mock(
        return_value=httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html"})
    )
    respx.get(f"{SITE}/internet").mock(
        return_value=httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html"})
    )

    crawler = SitemapCrawler(SITE, client=client)
    documents = list(crawler.documents())

    assert crawler.urls() == [SITE, f"{SITE}/internet"]
    assert not category.called
    assert [doc.url for doc in documents] == [SITE, f"{SITE}/internet"]
    assert documents[1].title == "Internet | Exemplo"
    assert documents[1].sections[0].heading == "Planos"
    assert documents[1].source == "sitemap"


@respx.mock
def test_sitemap_crawler_survives_a_broken_page(client):
    respx.get(f"{SITE}/sitemap_index.xml").mock(return_value=httpx.Response(200, text=PAGES))
    respx.get(f"{SITE}/").mock(return_value=httpx.Response(500))
    respx.get(f"{SITE}/internet").mock(
        return_value=httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html"})
    )

    documents = list(SitemapCrawler(SITE, client=client).documents())

    assert [doc.url for doc in documents] == [f"{SITE}/internet"]


def wp_item(item_id: int, slug: str, title: str, body: str) -> dict:
    return {
        "id": item_id,
        "link": f"{SITE}/{slug}/",
        "title": {"rendered": title},
        "content": {"rendered": body},
        "modified": "2025-01-14T15:33:49",
    }


@respx.mock
def test_wordpress_source_paginates_pages_and_posts(client):
    pages_route = respx.get(f"{SITE}/wp-json/wp/v2/pages")
    pages_route.mock(
        return_value=httpx.Response(
            200,
            json=[wp_item(1, "internet", "Internet", "<p>Planos de fibra.</p>")],
            headers={"X-WP-TotalPages": "1"},
        )
    )
    posts_route = respx.get(f"{SITE}/wp-json/wp/v2/posts")
    posts_route.side_effect = [
        httpx.Response(
            200,
            json=[wp_item(10, "blog/a", "Post A &amp; B", "<p>Texto A.</p>")],
            headers={"X-WP-TotalPages": "2"},
        ),
        httpx.Response(
            200,
            json=[wp_item(11, "blog/b", "Post B", "<p>Texto B.</p>")],
            headers={"X-WP-TotalPages": "2"},
        ),
    ]

    documents = list(WordPressRestSource(SITE, per_page=1, client=client).documents())

    assert [doc.url for doc in documents] == [
        f"{SITE}/internet",
        f"{SITE}/blog/a",
        f"{SITE}/blog/b",
    ]
    assert documents[1].title == "Post A & B"
    assert documents[1].text == "Texto A."
    assert documents[1].modified == "2025-01-14T15:33:49"
    assert {doc.source for doc in documents} == {"wordpress"}
    assert posts_route.calls[1].request.url.params["page"] == "2"


@respx.mock
def test_wordpress_source_skips_empty_items_and_failed_types(client):
    respx.get(f"{SITE}/wp-json/wp/v2/pages").mock(return_value=httpx.Response(403))
    respx.get(f"{SITE}/wp-json/wp/v2/posts").mock(
        return_value=httpx.Response(
            200,
            json=[wp_item(1, "blog/empty", "Empty", ""), wp_item(2, "blog/ok", "Ok", "<p>Ok</p>")],
            headers={"X-WP-TotalPages": "1"},
        )
    )

    documents = list(WordPressRestSource(SITE, client=client).documents())

    assert [doc.title for doc in documents] == ["Ok"]


def test_wordpress_json_with_leading_markup_is_still_parsed():
    from henet_kb.ingest.wordpress import parse_json_body

    body = '<style id="x">.a{color:red}</style>[{"id": 1}]'
    assert parse_json_body(body) == [{"id": 1}]
    with pytest.raises(ValueError):
        parse_json_body("<html>not json</html>")
