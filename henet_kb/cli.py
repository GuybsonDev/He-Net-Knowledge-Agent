import argparse
import json
import logging
import sys

from henet_kb.config import get_settings
from henet_kb.index import HybridIndex, OpenAIEmbedder
from henet_kb.ingest import SitemapCrawler, Source, WordPressRestSource, chunk_document


def build_index() -> HybridIndex:
    settings = get_settings()
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    return HybridIndex.from_settings(settings, embedder)


def make_source(name: str, site_url: str, max_pages: int | None) -> Source:
    if name == "wordpress":
        return WordPressRestSource(site_url)
    return SitemapCrawler(site_url, max_pages=max_pages)


def ingest(source: Source, index: HybridIndex, reset: bool = False) -> dict[str, int]:
    if reset:
        index.clear()
    documents = 0
    chunks = 0
    for document in source.documents():
        pieces = chunk_document(document)
        chunks += index.replace_document_chunks(document.url, pieces)
        documents += 1
        logging.info("indexed %s (%d chunks)", document.url, len(pieces))
    return {"documents": documents, "chunks": chunks, "total_chunks": index.count()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="henet-kb")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = sub.add_parser("ingest", help="crawl a source and (re)build the index")
    ingest_cmd.add_argument("--source", choices=["sitemap", "wordpress"], default="sitemap")
    ingest_cmd.add_argument("--site-url", default=None)
    ingest_cmd.add_argument("--max-pages", type=int, default=None)
    ingest_cmd.add_argument("--reset", action="store_true", help="drop the collection first")
    ingest_cmd.add_argument(
        "--dry-run", action="store_true", help="list documents and chunk counts, do not index"
    )

    search_cmd = sub.add_parser("search", help="run a hybrid search and print the hits")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    if args.command == "ingest":
        source = make_source(args.source, args.site_url or settings.site_url, args.max_pages)
        if args.dry_run:
            total = 0
            for document in source.documents():
                pieces = chunk_document(document)
                total += len(pieces)
                print(f"{len(pieces):4d} chunks  {document.url}  ({document.title})")
            print(json.dumps({"total_chunks": total}))
            return 0
        index = build_index()
        summary = ingest(source, index, reset=args.reset)
        summary["embedding_tokens"] = index.embedder.total_tokens  # type: ignore[attr-defined]
        print(json.dumps(summary))
        return 0

    if args.command == "search":
        for hit in build_index().search(args.query, top_k=args.top_k):
            print(f"{hit.score:.4f}  {hit.url}  [{hit.section}]\n    {hit.text[:160]}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
