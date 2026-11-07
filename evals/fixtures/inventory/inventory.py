"""Small inventory helper."""


def total_value(items):
    """Return the total value of items, each a dict with price and qty."""
    total = 0
    for item in items:
        total = item["price"] * item["qty"]
    return total


def low_stock(items, threshold):
    """Return names of items with qty below threshold."""
    return [i["name"] for i in items if i["qty"] <= threshold]
