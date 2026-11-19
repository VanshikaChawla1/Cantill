# DECISIONS.md

## Data Findings & Strategic Engineering Decisions

### 1. Dual-System Deduplication & Reconciliation (Company A)
- **Finding:** Company A has 37,029 ERP goods-receipt lines and 28,520 Invoice Tool export lines. Both systems share purchases through PO references, but tool PO references are inconsistently formatted (e.g. `4510085877/10.0`, `PO 4510109330`, `PO#4510059209`).
- **Decision:** Regex extraction normalizes all PO numbers into the standard 10-digit SAP format (`45\d{8}`). When an Invoice Tool record references a PO already present in the ERP goods receipts, the ERP line is preserved as the single source of truth, and the tool record is flagged with exclusion reason `DUPLICATE` (net spend set to 0.00).
- **Internal Tool Duplicates:** 831 records in the Invoice Tool export represent duplicate transmissions of the same invoice number, vendor, and gross amount. The first appearance is evaluated; subsequent identical records are marked as `DUPLICATE`.

### 2. Status Exclusions (Company A)
- **Finding:** The Invoice Tool export includes various statuses: `Paid` (19,837), `Approved` (8,180), `Cancelled` (188), `On Hold` (172), and `Rejected` (143).
- **Decision:** Following Section 4 & 5 of the brief:
  - Invoices with status `Cancelled` are excluded with reason `CANCELLED`.
  - Invoices with status `On Hold`, `Rejected`, `Submitted`, or `Disputed` are excluded with reason `NOT_APPROVED`.
  - Invoices with status `Approved` or `Paid` are included (unless matching an ERP receipt or duplicate transmission).

### 3. Non-Procurement & Tax Exclusions (Company B)
- **Finding:** In Company B, the 3,000 seed labels reveal 450 exclusions labeled `NON_PROCUREMENT`. Inspection shows that these correspond strictly to two patterns:
  1. Intercompany transactions: Identified by GL account descriptions starting with `IC`, `I/C`, or `IC ` (e.g., `IC Payables`, `I/C Payable`, `IC06`, `IC05`).
  2. Tax lines: Identified by line descriptions or GL descriptions mentioning taxes (e.g., `State & Local Taxes -`, `Sales Tax Expense -`, `GST Input Tax -`).
- **Decision:** All records matching intercompany accounts or explicit tax descriptions are assigned `include = 0` with exclusion reason `NON_PROCUREMENT`. This achieved 100.00% precision and recall on the 3,000 seed labels.

### 4. Currency Conversion & Exchange Rate Lookups (Company B)
- **Finding:** Company B invoices span 25 currencies across 8 ERPs. 20,023 records lack an explicit exchange rate (`currency_rate`).
- **Decision:** When `currency == "USD"`, the amount is taken directly. If an explicit positive `currency_rate` is provided, USD spend is calculated as `line_amount / currency_rate`. For missing rates, the monthly rate from `fx_rates_usd.csv` is matched using the invoice date month (`YYYY-MM`), falling back to posting date month if invoice date is unparseable.

### 5. Supplier and Item Deduplication
- **Finding:** Supplier names contain varying punctuation, casing, and legal suffix variations (e.g. `(PTY) LTD`, `PTY LTD`, `LTD`, `CC`, `GMBH`, `LLC`). Items in ERP have explicit material numbers for materials, but blank material numbers for free text and services.
- **Decision:**
  - Suppliers: Stripped of legal suffixes and punctuation, lowercased, and mapped to a canonical key format `SG-<md5_hash>`.
  - Items: If an explicit material number exists, it is hashed to create `IG-<md5_hash>`. If missing (e.g., services, free-text), the normalized line description is hashed. If description is blank, item key is left empty as seen in ground-truth labels.

### 6. Zero-Shot Classification Strategy for Unseen Categories
- **Finding:** Category trees differ between companies (Company A has 40 leaf categories; Company B has 40 leaf categories across 9 L1s; Company C is from a new industry with up to 4 levels). Ground-truth labels are unavailable for new companies.
- **Decision:** Built a zero-shot semantic matching engine that dynamically ingests any hierarchical taxonomy CSV at runtime. It combines sentence-transformers semantic embeddings (`all-MiniLM-L6-v2`) with character/word n-gram TF-IDF representations. Confidence scores reflect normalized cosine similarities.

