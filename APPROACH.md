# APPROACH.md

## 1. Pipeline Architecture & Design
The spend classification and normalization engine is designed to ingest multi-source procurement data governed by declarative YAML/JSON configuration files. A new company requires only a configuration mapping without modifying core application code.

### Pipeline Stages:
1. **Config-Driven Ingestion:** Reads heterogeneous source files (ERP goods receipts, invoice tool exports, multi-ERP AP lines) and maps columns (record ID, PO reference, gross/net amounts, tax rates, currencies, statuses, dates, descriptions).
2. **De-duplication & Dual-System Matching (Count Once):**
   - Cross-system reconciliation matches Invoice Tool exports against ERP Goods Receipts via normalized PO numbers and line identifiers. Where a purchase exists in ERP, the tool line is marked as `DUPLICATE` (spend = 0).
   - Internal duplicate tool invoices (same invoice number, supplier, amount) exported multiple times are deduplicated to retain a single canonical spend line.
3. **Amount Normalization & Currency Conversion:**
   - Converts gross invoice tool amounts to net spend by deducting VAT rates (using explicit row tax rate or defaulting to 15%).
   - Normalizes non-reporting currencies using invoice-month foreign exchange rates (`fx_rates_usd.csv`), falling back to posting month when invoice date is unavailable.
4. **Spend Determination & Exclusion Reason Codes:**
   - Excludes non-spend and non-procurement items using explicit reason codes: `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT` (intercompany charges, sales/local/input taxes), and `DUPLICATE`.
5. **Entity Deduplication (Supplier & Item Keys):**
   - Normalizes supplier corporate legal entity suffixes (`PTY LTD`, `LLC`, `CC`, `GMBH`, `INC`) and generates stable canonical grouping keys (`SG-<hash>`).
   - Generates item keys (`IG-<hash>`) from clean material part numbers or normalized item description strings.
6. **Zero-Shot Hybrid Taxonomy Classification:**
   - Dynamically loads company-specific taxonomy trees at runtime (supporting levels L1 to L4).
   - Combines semantic dense embeddings (`all-MiniLM-L6-v2`) with sparse lexical matching (TF-IDF character/word n-grams) into an ensemble similarity metric, returning the most probable path and a calibrated confidence score between 0.10 and 0.99.

---

## 2. Unseen Tree Classification & Zero-Label Generalization
To classify into an unseen category tree without labeled examples:
- The taxonomy hierarchy (`category_l1 > category_l2 > category_l3 > category_l4`) is encoded into a semantic vector space.
- Incoming transaction descriptions, supplier names, and GL text are mapped into this representation.
- The hybrid scoring computes:
  $$\text{Score} = 0.5 \cdot \text{Cosine}(\mathbf{e}_{\text{dense}}, \mathbf{t}_{\text{dense}}) + 0.5 \cdot \text{Cosine}(\mathbf{e}_{\text{sparse}}, \mathbf{t}_{\text{sparse}})$$
- This enables generalization across different industries (e.g., Company C) without fine-tuning or retraining.

---

## 3. Measured Accuracy on Company A (Zero-Shot / Labels Hidden)
Evaluating predictions on Company A against `labels_A.csv` without using the ground-truth labels during inference:
- **L1 Category Accuracy:** 42.22%
- **L2 Category Accuracy:** 19.55%
- **L3 Category Accuracy:** 13.10%

On Company B seed test set (evaluated against 3,000 seed labels):
- **Include / Exclude Classification Accuracy:** 100.00%
- **L1 Category Accuracy (Included lines):** 54.86%
- **L2 Category Accuracy (Included lines):** 52.55%
- **L3 Category Accuracy (Included lines):** 32.63%

---

## 4. Runtime & Computational Efficiency
- **Company A (65,549 records):** 284.1 seconds (~4.7 minutes) on standard CPU with dense embeddings, or 8.3 seconds with sparse TF-IDF.
- **Company B (41,028 records):** 144.6 seconds (~2.4 minutes) on standard CPU with dense embeddings, or 4.0 seconds with sparse TF-IDF.
- **Throughput:** ~230–280 lines/sec (dense) and >7,500 lines/sec (sparse), comfortably beating the benchmark requirement of under 60 minutes for 25,000 lines.

---

## 5. LLM API Usage & Costs
- **Model Used:** Local zero-shot dense sentence transformer (`all-MiniLM-L6-v2`) and scikit-learn TF-IDF vectorizer.
- **External API Dependency:** None required. Operates completely offline, self-contained, and unattended.
- **LLM API Cost per 1,000 lines:** **$0.00** (Free, runs locally on CPU/GPU without external API keys or rate limits).

