import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Short Portuguese stopword list. BM25 already handles frequency, this only drops the
# words that show up in nearly every chunk.
STOPWORDS = frozenset(
    "a o e de da do das dos em um uma para com por que se no na nos nas ao aos as os "
    "ou mais como seu sua seus suas ser tem ter pelo pela sao sobre entre ate".split()
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    words = TOKEN_RE.findall(strip_accents(text).lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]
