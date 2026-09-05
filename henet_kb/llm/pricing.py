# USD per million tokens, input then output. Update when the price lists change.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD. Unknown models cost zero rather than failing the request."""
    prices = PRICES.get(model)
    if prices is None:
        # Dated snapshots share the base model price. Longest prefix wins, otherwise
        # the mini variant would resolve to the full model.
        matches = [name for name in PRICES if model.startswith(name)]
        if matches:
            prices = PRICES[max(matches, key=len)]
    if prices is None:
        return 0.0
    input_price, output_price = prices
    return round((input_tokens * input_price + output_tokens * output_price) / 1_000_000, 6)
