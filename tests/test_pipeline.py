import os
import unittest
import pandas as pd
from src.pipeline import (
    clean_text,
    clean_supplier_name,
    extract_po_number,
    clean_numeric_amount,
    clean_vat_rate,
    SpendPipeline
)


class TestSpendPipeline(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("FAN ELEC; 22KW!!"), "fan elec 22kw")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")

    def test_clean_supplier_name(self):
        self.assertEqual(clean_supplier_name("ACME ENGINEERING (PTY) LTD"), "ACME ENGINEERING")
        self.assertEqual(clean_supplier_name("DUNTON (PTY) LTD"), "DUNTON")
        self.assertEqual(clean_supplier_name("CALWOOD LLC"), "CALWOOD")
        self.assertEqual(clean_supplier_name("WEXVEX CC"), "WEXVEX")

    def test_extract_po_number(self):
        self.assertEqual(extract_po_number("4510085877/10.0"), "4510085877")
        self.assertEqual(extract_po_number("PO 4510109330"), "4510109330")
        self.assertEqual(extract_po_number("PO#4510059209"), "4510059209")
        self.assertIsNone(extract_po_number("123456"))
        self.assertIsNone(extract_po_number(None))

    def test_clean_numeric_amount(self):
        self.assertEqual(clean_numeric_amount("1,234.56"), 1234.56)
        self.assertEqual(clean_numeric_amount("-500.00"), -500.0)
        self.assertEqual(clean_numeric_amount(100), 100.0)
        self.assertEqual(clean_numeric_amount(None), 0.0)

    def test_clean_vat_rate(self):
        self.assertEqual(clean_vat_rate("15%"), 0.15)
        self.assertEqual(clean_vat_rate("0.15"), 0.15)
        self.assertEqual(clean_vat_rate("15"), 0.15)
        self.assertEqual(clean_vat_rate(None, default_rate=0.15), 0.15)
        self.assertEqual(clean_vat_rate("0"), 0.0)

    def test_submission_a_schema(self):
        path = "submission/vanshika__A.csv"
        self.assertTrue(os.path.exists(path), f"{path} must exist")
        df = pd.read_csv(path, nrows=50)
        expected_cols = [
            "record_id", "include", "exclusion_reason", "supplier_key",
            "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"
        ]
        self.assertEqual(list(df.columns), expected_cols)
        self.assertTrue(set(df["include"].unique()).issubset({0, 1}))

    def test_submission_b_schema(self):
        path = "submission/vanshika__B.csv"
        self.assertTrue(os.path.exists(path), f"{path} must exist")
        df = pd.read_csv(path, nrows=50)
        expected_cols = [
            "record_id", "include", "exclusion_reason", "supplier_key",
            "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"
        ]
        self.assertEqual(list(df.columns), expected_cols)
        self.assertTrue(set(df["include"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
