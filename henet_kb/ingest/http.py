import httpx

USER_AGENT = "henet-kb/2.0 (+https://github.com/GuybsonDev/Treinamento-IA-Base-Henet)"


def make_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )
