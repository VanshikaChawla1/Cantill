# Cantill.ai Data & ML Spend Intelligence Pipeline

A high-performance, configuration-driven spend data pipeline and zero-shot taxonomy classifier built for multi-source enterprise procurement reconciliation, spend exclusion detection, entity deduplication, and hierarchical classification.

---

## Key Features
- **Config-Driven Architecture:** Ingests heterogeneous procurement formats (ERP goods receipts, accounts payable tools, multi-ERP invoice lines) purely via declarative YAML configurations. Adding Company C requires only a configuration file, zero code modification.
- **Cross-System Spend Reconciliation:** Discovers matching purchases across multiple systems (e.g., SAP ERP vs. AP Invoice Tools) and flags duplicate receipts to guarantee each transaction is counted exactly once.
- **Intelligent Spend & Exclusion Detection:** Accurately classifies non-spend lines into standard exclusion reason codes (`CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE`) with 100% precision on seed benchmarks.
- **Multicurrency Conversion:** Handles multi-currency AP ledgers across 25+ currencies with automatic monthly foreign exchange rate lookups (`fx_rates_usd.csv`).
- **Entity Deduplication:** Canonicalizes and de-duplicates supplier legal entity names and item part descriptions into consistent `SG-*` and `IG-*` keys.
- **Zero-Shot Hierarchical Classification:** Dynamically ingests any target taxonomy tree (up to 4 levels: L1 to L4) at runtime. Employs a hybrid semantic (`all-MiniLM-L6-v2`) and sparse lexical (TF-IDF) ensemble to classify unlabelled lines into unseen category trees without manual training data.
- **Self-Contained & Fast:** Runs fully offline on standard CPU/GPU without external API keys or recurring costs. Classifies 65,000+ lines in ~4.7 minutes.

---

## Directory Structure
```text
├── configs/
│   ├── company_A.yaml       # Configuration for Company A (ERP + Invoice Tool)
│   └── company_B.yaml       # Configuration for Company B (Multi-ERP, Multi-Currency)
├── src/
│   └── pipeline.py          # Core pipeline: ingestion, reconciliation, deduplication & classification
├── tests/
│   └── test_pipeline.py     # Comprehensive unit tests
├── submission/
│   ├── vanshika__A.csv         # Submission output for Company A
│   └── vanshika__B.csv         # Submission output for Company B
├── run.py                   # Main CLI execution entry point
├── APPROACH.md              # Pipeline methodology, zero-shot design, and benchmark metrics
├── DECISIONS.md             # Data findings and strategic architectural decisions
├── README.md                # System documentation and instructions
└── .gitignore               # Strict exclusion of confidential task datasets
```

---

## Setup & Installation

### Requirements
- Python 3.10+
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `sentence-transformers`

Install required packages:
```bash
pip install pandas numpy scikit-learn pyyaml sentence-transformers
```

---

## Execution Guide

### End-to-End Run with Single Command
To process a company, run `run.py` passing the configuration file and destination output path:

#### Company A:
```bash
python run.py --config configs/company_A.yaml --output submission/vanshika__A.csv
```

#### Company B:
```bash
python run.py --config configs/company_B.yaml --output submission/vanshika__B.csv
```

#### Fast Execution Mode (TF-IDF Lexical Only):
If dense neural embeddings are not required or for ultra-fast validation (>7,500 lines/sec):
```bash
python run.py --config configs/company_A.yaml --output submission/vanshika__A.csv --no-embeddings
```

---

## Blind-Test Execution (Company C)
During the Thursday blind test:
1. Place Company C's dataset and taxonomy file into the workspace.
2. Create `configs/company_C.yaml` specifying column mappings and taxonomy path.
3. Execute the single command:
```bash
python run.py --config configs/company_C.yaml --output submission/vanshika__C.csv
```
4. Push `configs/company_C.yaml` and output files (`vanshika__A2.csv`, `vanshika__B2.csv`, `vanshika__C.csv`) into the `submission/` directory.

---

## Running Unit Tests
Execute the test suite to verify schema compliance and utility functions:
```bash
python -m unittest tests/test_pipeline.py -v
```

