import pytest

from henet_kb.index.hybrid import HybridIndex
from henet_kb.index.text import tokenize
from henet_kb.ingest.base import Document, Section
from henet_kb.ingest.chunking import chunk_document
from tests.fakes import HashEmbedder


def build_index() -> HybridIndex:
    index = HybridIndex.ephemeral(HashEmbedder())
    documents = [
        Document(
            url="https://henet.com.br/internet",
            title="Internet",
            sections=[Section("Planos", "Plano de fibra optica com 500 mega e Wi-Fi 6 incluso.")],
        ),
        Document(
            url="https://henet.com.br/tv",
            title="TV",
            sections=[Section("Canais", "Pacote de TV com canais de esporte e filmes em HD.")],
        ),
        Document(
            url="https://henet.com.br/fale-conosco",
            title="Fale Conosco",
            sections=[Section("Contato", "Atendimento pelo WhatsApp 75 4003-1234 e loja fisica.")],
        ),
    ]
    for document in documents:
        index.add_chunks(chunk_document(document))
    return index


def test_tokenize_normalizes_accents_and_stopwords():
    assert tokenize("A fibra ÓPTICA de 500 mega") == ["fibra", "optica", "500", "mega"]


def test_search_returns_hits_with_source_metadata():
    index = build_index()

    hits = index.search("qual a velocidade da fibra", top_k=2)

    assert hits[0].url == "https://henet.com.br/internet"
    assert hits[0].title == "Internet"
    assert hits[0].section == "Planos"
    assert hits[0].chunk_index == 0
    assert hits[0].score > 0


def test_keyword_match_surfaces_exact_terms():
    index = build_index()

    hits = index.search("WhatsApp", top_k=1)

    assert hits[0].url == "https://henet.com.br/fale-conosco"


def test_results_are_fused_without_duplicates():
    index = build_index()

    hits = index.search("canais de esporte em HD", top_k=5)

    ids = [hit.chunk_id for hit in hits]
    assert len(ids) == len(set(ids))
    assert hits[0].url == "https://henet.com.br/tv"
    assert len(hits) <= 5


def test_replace_document_chunks_swaps_old_content():
    index = build_index()
    updated = Document(
        url="https://henet.com.br/tv",
        title="TV",
        sections=[Section("Canais", "Pacote de TV com canais infantis e documentarios.")],
    )

    index.replace_document_chunks(updated.url, chunk_document(updated))

    assert index.count() == 3
    assert index.search("esporte", top_k=1)[0].url != "https://henet.com.br/tv" or not index.search(
        "esporte", top_k=1
    )
    assert index.search("documentarios", top_k=1)[0].url == "https://henet.com.br/tv"


def test_empty_index_returns_nothing():
    index = HybridIndex.ephemeral(HashEmbedder())

    assert index.search("qualquer coisa") == []


@pytest.mark.parametrize("top_k", [1, 3])
def test_top_k_is_respected(top_k):
    assert len(build_index().search("plano", top_k=top_k)) <= top_k
