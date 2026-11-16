"""
Spend Ingestion, Normalization, Deduplication, and Classification Pipeline
Cantill Data & ML Intern Task
"""

import os
import re
import math
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def clean_text(text: Any) -> str:
    """Normalize text by stripping punctuation, extra whitespace, and lowercasing."""
    if pd.isna(text) or text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return ' '.join(s.split())


def clean_supplier_name(name: Any) -> str:
    """Normalize supplier company names, stripping legal entity suffixes."""
    if pd.isna(name) or name is None:
        return ""
    s = str(name).upper().strip()
    # Strip common corporate suffixes
    suffixes = [
        r'\(PTY\)\s*LTD', r'PTY\s*LTD', r'\(PTY\)', r'PTY',
        r'LTD\.?', r'INC\.?', r'CORP\.?', r'LLC\.?', r'CC\.?',
        r'GMBH\.?', r'BV\.?', r'SA\.?', r'CO\.?'
    ]
    for suf in suffixes:
        s = re.sub(r'\b' + suf + r'\b', '', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = ' '.join(s.split())
    return s.strip()


def extract_po_number(val: Any) -> Optional[str]:
    """Extract standard PO number pattern (e.g. 45 followed by 8 digits)."""
    if pd.isna(val) or val is None:
        return None
    m = re.search(r'(45\d{8})', str(val))
    return m.group(1) if m else None


def clean_numeric_amount(val: Any) -> float:
    """Parse numeric amount from currency/string format."""
    if pd.isna(val) or val is None:
        return 0.0
    s = str(val).replace(',', '').strip()
    try:
        return float(s)
    except:
        return 0.0


def clean_vat_rate(val: Any, default_rate: float = 0.15) -> float:
    """Parse VAT percentage or fraction."""
    if pd.isna(val) or val is None:
        return default_rate
    s = str(val).replace('%', '').strip()
    try:
        v = float(s)
        if v > 1.0:
            v = v / 100.0
        return v
    except:
        return default_rate


class SpendPipeline:
    def __init__(self, config: Dict[str, Any], use_local_embeddings: bool = True):
        self.config = config
        self.company_id = config.get("company_id", "")
        self.reporting_currency = config.get("reporting_currency", "USD")
        self.use_local_embeddings = use_local_embeddings
        self.embed_model = None
        if self.use_local_embeddings:
            try:
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            except Exception as e:
                print(f"Notice: local sentence transformer could not be loaded: {e}. Falling back to TF-IDF.")
                self.embed_model = None

    def load_taxonomy(self) -> pd.DataFrame:
        tax_cfg = self.config["taxonomy"]
        df_tax = pd.read_csv(tax_cfg["path"])
        level_cols = tax_cfg["level_cols"]
        # Standardize level cols
        df_tax["full_path"] = df_tax[level_cols].apply(
            lambda row: " > ".join([str(x) for x in row if pd.notna(x) and str(x).strip() != ""]),
            axis=1
        )
        return df_tax

    def load_fx_rates(self) -> Dict[Tuple[str, str], float]:
        fx_cfg = self.config.get("fx_rates")
        if not fx_cfg:
            return {}
        df_fx = pd.read_csv(fx_cfg["path"])
        curr_col = fx_cfg["currency_col"]
        month_col = fx_cfg["month_col"]
        rate_col = fx_cfg["rate_col"]
        return df_fx.set_index([curr_col, month_col])[rate_col].to_dict()

    def process_company_A(self) -> pd.DataFrame:
        """Process Company A with ERP goods receipts and Invoice Tool export."""
        tax_df = self.load_taxonomy()
        sources = {s["id"]: s for s in self.config["sources"]}
        erp_cfg = sources["erp"]
        tool_cfg = sources["tool"]

        # Read ERP
        erp_df = pd.read_csv(erp_cfg["path"], low_memory=False)
        erp_df["raw_record_id"] = erp_df[erp_cfg["id_col"]].astype(str)
        erp_df["record_id"] = erp_cfg.get("prefix", "ERP:") + erp_df["raw_record_id"]
        erp_df["clean_po"] = erp_df[erp_cfg["po_col"]].apply(extract_po_number)
        erp_df["spend"] = erp_df[erp_cfg["amount_col"]].apply(clean_numeric_amount)
        erp_df["spend"] = erp_df["spend"].apply(lambda x: max(0.0, x))
        erp_df["include"] = 1
        erp_df["exclusion_reason"] = ""

        # Read Tool
        tool_df = pd.read_csv(tool_cfg["path"], low_memory=False)
        tool_df["raw_record_id"] = tool_df[tool_cfg["id_col"]].astype(str)
        tool_df["record_id"] = tool_cfg.get("prefix", "TOOL:") + tool_df["raw_record_id"]
        tool_df["clean_po"] = tool_df[tool_cfg["po_col"]].apply(extract_po_number)

        # Tool amount net of VAT
        gross = tool_df[tool_cfg["amount_col"]].apply(clean_numeric_amount)
        vat = tool_df[tool_cfg["vat_col"]].apply(lambda x: clean_vat_rate(x, tool_cfg.get("default_vat_rate", 0.15)))
        tool_df["net_spend"] = gross / (1.0 + vat)

        # Status exclusions
        status_col = tool_cfg["status_col"]
        tool_df["include"] = 1
        tool_df["exclusion_reason"] = ""

        # Rule: Status must be Approved or Paid
        for idx, row in tool_df.iterrows():
            st = str(row[status_col]).strip()
            if st == "Cancelled":
                tool_df.at[idx, "include"] = 0
                tool_df.at[idx, "exclusion_reason"] = "CANCELLED"
            elif st in ["On Hold", "Rejected", "Submitted", "Disputed"]:
                tool_df.at[idx, "include"] = 0
                tool_df.at[idx, "exclusion_reason"] = "NOT_APPROVED"

        # Rule: Count once - match records across systems
        erp_pos = set(erp_df["clean_po"].dropna())
        # Rule: Tool invoice exported twice - count it once
        # Check duplicate tool invoices
        seen_tool_invoices = set()
        for idx, row in tool_df.iterrows():
            if tool_df.at[idx, "include"] == 0:
                continue
            inv_key = (
                str(row.get("invoice_number", "")).strip(),
                str(row.get("supplier_name_as_captured", "")).strip(),
                round(clean_numeric_amount(row.get("gross_amount", 0)), 2)
            )
            if inv_key in seen_tool_invoices:
                tool_df.at[idx, "include"] = 0
                tool_df.at[idx, "exclusion_reason"] = "DUPLICATE"
                continue
            seen_tool_invoices.add(inv_key)

            # Check if PO exists in ERP
            po = row["clean_po"]
            if po and po in erp_pos:
                tool_df.at[idx, "include"] = 0
                tool_df.at[idx, "exclusion_reason"] = "DUPLICATE"

        # Final spend calculation
        tool_df["spend"] = tool_df.apply(lambda r: r["net_spend"] if r["include"] == 1 else 0.0, axis=1)

        # Combine ERP and Tool
        # Deduplicate suppliers & items
        # For ERP:
        erp_df["clean_supp_name"] = erp_df[erp_cfg["supplier_name_col"]].apply(clean_supplier_name)
        tool_df["clean_supp_name"] = tool_df[tool_cfg["supplier_name_col"]].apply(clean_supplier_name)

        # Build supplier keys
        # Use MD5 hash of canonical supplier name
        def get_supp_key(name):
            if not name:
                return "SUPP_UNKNOWN"
            h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
            return f"SG-{h}"

        erp_df["supplier_key"] = erp_df["clean_supp_name"].apply(get_supp_key)
        tool_df["supplier_key"] = tool_df["clean_supp_name"].apply(get_supp_key)

        # Build item keys
        # If material_number exists, use it; else hash of description
        def get_item_key(mat_num, desc):
            if pd.notna(mat_num) and str(mat_num).strip() != "" and str(mat_num).strip().lower() != "nan":
                m_str = str(mat_num).strip().upper()
                h = hashlib.md5(m_str.encode("utf-8")).hexdigest()[:7]
                return f"IG-{h}"
            c_desc = clean_text(desc)
            if c_desc:
                h = hashlib.md5(c_desc.encode("utf-8")).hexdigest()[:7]
                return f"IG-{h}"
            return ""

        erp_df["item_key"] = erp_df.apply(
            lambda r: get_item_key(r.get(erp_cfg["item_id_col"]), r.get(erp_cfg["description_col"]) or r.get(erp_cfg.get("service_description_col"))),
            axis=1
        )
        tool_df["item_key"] = tool_df.apply(
            lambda r: get_item_key(None, r.get(tool_cfg["description_col"])),
            axis=1
        )

        # Prepare unified text for classification
        erp_df["class_text"] = erp_df.apply(
            lambda r: (str(r.get(erp_cfg["description_col"]) or "") + " " + str(r.get(erp_cfg.get("service_description_col")) or "") + " " + str(r.get(erp_cfg["supplier_name_col"]) or "")).strip(),
            axis=1
        )
        tool_df["class_text"] = tool_df.apply(
            lambda r: (str(r.get(tool_cfg["description_col"]) or "") + " " + str(r.get(tool_cfg["supplier_name_col"]) or "")).strip(),
            axis=1
        )

        combined = pd.concat([erp_df, tool_df], ignore_index=True)

        # Classify lines
        preds = self.classify_texts(combined["class_text"].tolist(), tax_df)
        combined["l1"] = preds["l1"]
        combined["l2"] = preds["l2"]
        combined["l3"] = preds["l3"]
        combined["l4"] = preds.get("l4", "")
        combined["confidence"] = preds["confidence"]

        # Blank categories for excluded rows
        for idx in combined[combined["include"] == 0].index:
            combined.at[idx, "l1"] = ""
            combined.at[idx, "l2"] = ""
            combined.at[idx, "l3"] = ""
            combined.at[idx, "l4"] = ""

        # Format output
        output = pd.DataFrame({
            "record_id": combined["record_id"],
            "include": combined["include"],
            "exclusion_reason": combined["exclusion_reason"],
            "supplier_key": combined["supplier_key"],
            "item_key": combined["item_key"],
            "spend": combined["spend"].round(2),
            "l1": combined["l1"],
            "l2": combined["l2"],
            "l3": combined["l3"],
            "l4": combined["l4"],
            "confidence": combined["confidence"].round(2)
        })
        return output

    def process_company_B(self) -> pd.DataFrame:
        """Process Company B with multi-ERP, multi-currency invoice lines."""
        tax_df = self.load_taxonomy()
        fx_map = self.load_fx_rates()
        sources = {s["id"]: s for s in self.config["sources"]}
        ap_cfg = sources["ap_lines"]

        df = pd.read_csv(ap_cfg["path"], low_memory=False)
        id_col = ap_cfg["id_col"]
        df["record_id"] = ap_cfg.get("prefix", "") + df[id_col].astype(str)

        # Exclusions: Intercompany & Taxes
        gl_desc_col = ap_cfg.get("gl_description_col", "gl_account_description")
        desc_col = ap_cfg.get("description_col", "line_description")
        df["include"] = 1
        df["exclusion_reason"] = ""

        for idx, row in df.iterrows():
            gl_desc = str(row.get(gl_desc_col) or "").strip()
            desc = str(row.get(desc_col) or "").strip()

            # Intercompany check
            if re.search(r'^(IC|I/C|IC )', gl_desc, re.IGNORECASE):
                df.at[idx, "include"] = 0
                df.at[idx, "exclusion_reason"] = "NON_PROCUREMENT"
                continue

            # Taxes check
            if re.search(r'tax', desc, re.IGNORECASE) or re.search(r'tax', gl_desc, re.IGNORECASE):
                df.at[idx, "include"] = 0
                df.at[idx, "exclusion_reason"] = "NON_PROCUREMENT"
                continue

        # Currency conversion
        amt_col = ap_cfg["amount_col"]
        curr_col = ap_cfg["currency_col"]
        rate_col = ap_cfg["rate_col"]
        date_col = ap_cfg["date_col"]
        fallback_date_col = ap_cfg.get("fallback_date_col")

        def convert_row_spend(row):
            if row["include"] == 0:
                return 0.0
            amt = clean_numeric_amount(row.get(amt_col))
            curr = str(row.get(curr_col) or "").strip().upper()
            rate = row.get(rate_col)

            if curr == "USD":
                return amt
            if pd.notna(rate) and rate > 0:
                return amt / float(rate)

            # FX rates fallback by invoice month or posting month
            inv_date = str(row.get(date_col) or "")[:7]
            if (curr, inv_date) in fx_map:
                return amt / fx_map[(curr, inv_date)]

            post_date = str(row.get(fallback_date_col) or "")[:7]
            if (curr, post_date) in fx_map:
                return amt / fx_map[(curr, post_date)]

            # Default if rate missing
            return amt

        df["spend"] = df.apply(convert_row_spend, axis=1)

        # Supplier and Item keys
        vendor_col = ap_cfg.get("vendor_id_col", "vendor_id")
        item_col = ap_cfg.get("item_id_col", "material_number")

        def get_b_supp_key(v_id):
            if pd.isna(v_id) or not str(v_id).strip():
                return "SUPP_UNKNOWN"
            h = hashlib.md5(str(v_id).strip().upper().encode("utf-8")).hexdigest()[:6]
            return f"SG-{h}"

        def get_b_item_key(m_id, desc):
            if pd.notna(m_id) and str(m_id).strip() != "" and str(m_id).strip().lower() != "nan":
                h = hashlib.md5(str(m_id).strip().upper().encode("utf-8")).hexdigest()[:7]
                return f"IG-{h}"
            c_desc = clean_text(desc)
            if c_desc:
                h = hashlib.md5(c_desc.encode("utf-8")).hexdigest()[:7]
                return f"IG-{h}"
            return ""

        df["supplier_key"] = df[vendor_col].apply(get_b_supp_key)
        df["item_key"] = df.apply(lambda r: get_b_item_key(r.get(item_col), r.get(desc_col)), axis=1)

        # Classify lines
        df["class_text"] = df.apply(
            lambda r: (str(r.get(desc_col) or "") + " " + str(r.get(gl_desc_col) or "")).strip(),
            axis=1
        )

        preds = self.classify_texts(df["class_text"].tolist(), tax_df)
        df["l1"] = preds["l1"]
        df["l2"] = preds["l2"]
        df["l3"] = preds["l3"]
        df["l4"] = preds.get("l4", "")
        df["confidence"] = preds["confidence"]

        # Blank categories for excluded rows
        for idx in df[df["include"] == 0].index:
            df.at[idx, "l1"] = ""
            df.at[idx, "l2"] = ""
            df.at[idx, "l3"] = ""
            df.at[idx, "l4"] = ""

        # Format output
        output = pd.DataFrame({
            "record_id": df["record_id"],
            "include": df["include"],
            "exclusion_reason": df["exclusion_reason"],
            "supplier_key": df["supplier_key"],
            "item_key": df["item_key"],
            "spend": df["spend"].round(2),
            "l1": df["l1"],
            "l2": df["l2"],
            "l3": df["l3"],
            "l4": df["l4"],
            "confidence": df["confidence"].round(2)
        })
        return output

    def classify_texts(self, texts: List[str], tax_df: pd.DataFrame) -> Dict[str, Any]:
        """Hybrid Zero-Shot Semantic & Lexical Classifier into arbitrary taxonomy tree."""
        num_texts = len(texts)
        num_cats = len(tax_df)

        tax_paths = tax_df["full_path"].tolist()
        has_l4 = "category_l4" in tax_df.columns

        # Build TF-IDF representation
        all_tax_docs = []
        for _, row in tax_df.iterrows():
            doc = " ".join([clean_text(row[c]) for c in tax_df.columns if "category_" in c])
            all_tax_docs.append(doc)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10000)
        tfidf_tax = vectorizer.fit_transform(all_tax_docs)

        # Batch transform queries
        clean_queries = [clean_text(t) for t in texts]
        tfidf_queries = vectorizer.transform(clean_queries)
        tfidf_sims = cosine_similarity(tfidf_queries, tfidf_tax)

        final_sims = tfidf_sims

        if self.embed_model is not None:
            # Batch encode taxonomy and queries with sentence-transformers
            tax_embeddings = self.embed_model.encode(tax_paths, batch_size=64, normalize_embeddings=True)
            # Encode queries in batches
            batch_size = 512
            dense_sims = np.zeros((num_texts, num_cats), dtype=np.float32)
            for i in range(0, num_texts, batch_size):
                batch_texts = clean_queries[i:i+batch_size]
                batch_emb = self.embed_model.encode(batch_texts, batch_size=batch_size, normalize_embeddings=True)
                dense_sims[i:i+len(batch_texts)] = np.dot(batch_emb, tax_embeddings.T)

            # Ensemble combination: 0.5 * dense + 0.5 * sparse
            final_sims = 0.5 * dense_sims + 0.5 * tfidf_sims

        best_indices = np.argmax(final_sims, axis=1)
        best_scores = np.max(final_sims, axis=1)

        # Map to columns
        pred_l1, pred_l2, pred_l3, pred_l4 = [], [], [], []
        confidences = []

        for idx, cat_idx in enumerate(best_indices):
            row = tax_df.iloc[cat_idx]
            pred_l1.append(row.get("category_l1", ""))
            pred_l2.append(row.get("category_l2", ""))
            pred_l3.append(row.get("category_l3", ""))
            pred_l4.append(row.get("category_l4", "") if has_l4 else "")
            # Scale score to 0.1 - 0.99 range for confidence
            score = float(best_scores[idx])
            conf = min(0.99, max(0.10, score))
            confidences.append(conf)

        return {
            "l1": pred_l1,
            "l2": pred_l2,
            "l3": pred_l3,
            "l4": pred_l4,
            "confidence": np.array(confidences)
        }
