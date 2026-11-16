#!/usr/bin/env python3
"""
CLI entry point for Cantill Spend Data Pipeline and Classifier.
Usage:
    python run.py --config configs/company_A.yaml --output submission/vanshika__A.csv
    python run.py --config configs/company_B.yaml --output submission/vanshika__B.csv
"""

import argparse
import os
import sys
import yaml
import time
from src.pipeline import SpendPipeline


def main():
    parser = argparse.ArgumentParser(description="Cantill Spend Intelligence Pipeline")
    parser.add_argument("--config", required=True, help="Path to company YAML config file")
    parser.add_argument("--output", required=True, help="Path to write output CSV")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable local dense embeddings and use TF-IDF only")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    print(f"Loading config from {args.config}...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    start_time = time.time()
    pipeline = SpendPipeline(config, use_local_embeddings=not args.no_embeddings)

    company_id = config.get("company_id", "")
    print(f"Running pipeline for {company_id}...")

    if "company_A" in company_id:
        df_out = pipeline.process_company_A()
    elif "company_B" in company_id:
        df_out = pipeline.process_company_B()
    else:
        # Generic company (e.g. Company C in blind test)
        df_out = pipeline.process_company_B()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    elapsed = time.time() - start_time

    print(f"Successfully processed {len(df_out)} rows in {elapsed:.1f}s.")
    print(f"Output saved to: {args.output}")
    print(f"Spend summary: Included={df_out['include'].sum()}, Total Spend={df_out['spend'].sum():,.2f}")


if __name__ == "__main__":
    main()

