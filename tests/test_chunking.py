from henet_kb.ingest.base import Document, Section
from henet_kb.ingest.chunking import chunk_document, count_tokens


def make_document(sentences_per_section: int = 40) -> Document:
    sentence = "A He-Net oferece planos de internet por fibra em varias cidades da Bahia. "
    return Document(
        url="https://henet.com.br/internet",
        title="Internet",
        sections=[
            Section(heading="Planos", text=sentence * sentences_per_section),
            Section(heading="Cobertura", text="Atendemos Feira de Santana e regiao."),
        ],
    )


def test_chunks_respect_token_budget_and_keep_metadata():
    chunks = chunk_document(make_document(), max_tokens=120, overlap_tokens=20)

    assert len(chunks) > 3
    assert all(count_tokens(chunk.text) <= 120 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.url for chunk in chunks} == {"https://henet.com.br/internet"}
    assert {chunk.title for chunk in chunks} == {"Internet"}
    assert chunks[0].section == "Planos"
    assert chunks[-1].section == "Cobertura"
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_chunks_never_cross_section_boundaries():
    chunks = chunk_document(make_document(sentences_per_section=2), max_tokens=400)

    assert len(chunks) == 2
    assert "Feira de Santana" not in chunks[0].text
    assert "Feira de Santana" in chunks[1].text


def test_consecutive_chunks_overlap():
    chunks = chunk_document(make_document(), max_tokens=100, overlap_tokens=30)

    first_tail = chunks[0].text.split(". ")[-1].strip(". ")
    assert first_tail in chunks[1].text


def test_long_sentence_is_split_instead_of_dropped():
    long_sentence = "palavra " * 300
    document = Document(url="u", title="t", sections=[Section(heading="", text=long_sentence)])

    chunks = chunk_document(document, max_tokens=50, overlap_tokens=0)

    assert len(chunks) >= 6
    assert "".join(chunk.text for chunk in chunks).count("palavra") == 300


def test_empty_document_yields_no_chunks():
    assert chunk_document(Document(url="u", title="t")) == []


def test_chunk_ids_are_deterministic():
    first = chunk_document(make_document(), max_tokens=120)
    second = chunk_document(make_document(), max_tokens=120)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
