from henet_kb.ingest.clean import extract_title, html_to_document, html_to_sections

PAGE = """
<html><head><title>Internet - He-Net - Internet do seu jeito!</title>
<meta property="og:title" content="Internet - He-Net - Internet do seu jeito!"></head>
<body>
<header><nav><ul><li>Home</li><li>TV</li></ul></nav></header>
<div class="cookie-notice"><p>Utilizamos cookies para oferecer a melhor experiencia.</p></div>
<main>
  <h2>Planos de internet</h2>
  <p>Fibra optica com 500 mega de velocidade.</p>
  <ul><li><p>Wi-Fi 6 incluso</p></li><li>Instalacao gratis</li></ul>
  <h2>Cobertura</h2>
  <p>Atendemos Feira de Santana.</p>
  <script>var x = 1;</script>
</main>
<footer><p>He-Net 2024. Todos os direitos reservados.</p></footer>
</body></html>
"""


def test_sections_follow_headings_and_drop_chrome():
    sections = html_to_sections(PAGE)

    assert [section.heading for section in sections] == ["Planos de internet", "Cobertura"]
    assert "500 mega" in sections[0].text
    assert "Wi-Fi 6 incluso" in sections[0].text
    assert "Instalacao gratis" in sections[0].text
    joined = " ".join(section.text for section in sections)
    assert "cookies" not in joined
    assert "direitos reservados" not in joined
    assert "var x" not in joined


def test_nested_block_text_is_not_duplicated():
    sections = html_to_sections(PAGE)

    assert sections[0].text.count("Wi-Fi 6 incluso") == 1


def test_title_strips_site_suffix():
    assert extract_title(PAGE, "https://henet.com.br/internet") == "Internet"


def test_title_falls_back_to_url_path():
    url = "https://henet.com.br/monte-seu-combo"
    assert extract_title("<html></html>", url) == "Monte seu combo"
    assert extract_title("<html></html>", "https://henet.com.br/") == "Home"


def test_fragment_without_headings_becomes_single_section():
    fragment = "<p>Primeiro paragrafo do post.</p><p>Segundo paragrafo.</p>"

    sections = html_to_sections(fragment)

    assert len(sections) == 1
    assert sections[0].heading == ""
    assert "Segundo paragrafo" in sections[0].text


def test_document_from_wordpress_fragment_keeps_given_title():
    document = html_to_document(
        "<p>Conteudo</p>",
        url="https://henet.com.br/blog/x",
        source="wordpress",
        title="Titulo do post",
        modified="2025-01-14",
    )

    assert document.title == "Titulo do post"
    assert document.text == "Conteudo"
    assert document.source == "wordpress"
    assert document.modified == "2025-01-14"


def test_nested_chrome_elements_are_removed_without_errors():
    page = """<main>
    <div class="jet-menu"><ul class="menu"><li class="menu-item"><a>Home</a></li></ul></div>
    <div id="cookie-bar"><div class="inner"><p>Aceitar cookies</p></div></div>
    <h2>Conteudo</h2><p>Texto util.</p></main>"""

    sections = html_to_sections(page)

    assert [section.heading for section in sections] == ["Conteudo"]
    assert "cookies" not in sections[0].text
