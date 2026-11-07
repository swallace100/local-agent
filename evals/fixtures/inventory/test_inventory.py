import unittest

from inventory import low_stock, total_value

ITEMS = [
    {"name": "bolt", "price": 2.0, "qty": 10},
    {"name": "nut", "price": 1.5, "qty": 4},
]


class InventoryTests(unittest.TestCase):
    def test_total_value(self):
        self.assertEqual(total_value(ITEMS), 26.0)

    def test_low_stock_excludes_threshold(self):
        self.assertEqual(low_stock(ITEMS, 4), [])


if __name__ == "__main__":
    unittest.main()
