#!/usr/bin/env python3

import os
import json
import glob
import csv

# pull both of your sample folders and model name from config
from utils.config import INVOICES_DIR, GROUND_TRUTH_DIR as GT_DIR, OLLAMA_MODEL

from utils.pdf_utils         import pdf_to_png
from tesseract_ocr           import ocr_png_to_text
from layoutlmv3_png2txt     import layoutlm_image_to_text
from doctr_pdf2txt           import doctr_pdf_to_text
from utils.searchable_pdf    import extract_text_if_searchable
from utils.llm_utils         import ask_ollama_for_invoice_fields

# sanitize model name for filenames (replace ':' with '_')
MODEL_SAFE = OLLAMA_MODEL.replace(':', '_')

# CONFIGURE ▼
# Output filenames include sanitized model name only
OUTPUT_SUMMARY_CSV = f"benchmark_summary_{MODEL_SAFE}.csv"
OUTPUT_DETAIL_CSV  = f"benchmark_details_{MODEL_SAFE}.csv"
USE_SEARCHABLE     = False      # prefer embedded text when True
PIPELINES          = ["ocr", "layoutlm", "doctr"]
# ▲ end of user-tweakable section

def extract_text(pdf_path: str, pipeline: str, use_searchable: bool = False) -> str | None:
    """Run one of the three pipelines + optional searchable-PDF fallback."""
    if use_searchable:
        txt = extract_text_if_searchable(pdf_path)
        if txt:
            return txt

    if pipeline == "doctr":
        return doctr_pdf_to_text(pdf_path)

    # rasterize for ocr & layoutlm
    png = pdf_to_png(pdf_path)
    if pipeline == "ocr":
        return ocr_png_to_text(png)
    elif pipeline == "layoutlm":
        return layoutlm_image_to_text(png)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

def is_match(true, pred) -> bool:
    """Exact-match comparison, with simple null/number handling."""
    if true is None:
        return pred is None or str(pred).strip().lower() in ("", "none")
    if isinstance(true, (int, float)):
        try:
            return float(pred) == float(true)
        except:
            return False
    return str(pred).strip() == str(true).strip()

def main():
    inv_paths = sorted(glob.glob(os.path.join(INVOICES_DIR, "*.pdf")))
    if not inv_paths:
        raise RuntimeError(f"No PDFs found in {INVOICES_DIR!r}")

    # derive field list from the first ground-truth JSON
    first_base = os.path.basename(inv_paths[0]).rsplit(".", 1)[0]
    first_gt   = json.load(open(os.path.join(GT_DIR, f"{first_base}.json"), encoding="utf-8"))
    fields     = list(first_gt.keys())

    # prepare summary CSV
    summary_header = ["invoice", "pipeline"] + fields + ["accuracy"]
    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as sf:
        summary_writer = csv.DictWriter(sf, fieldnames=summary_header)
        summary_writer.writeheader()

        # prepare detailed CSV
        detail_header = ["invoice", "pipeline", "field", "expected", "predicted", "match"]
        with open(OUTPUT_DETAIL_CSV, "w", newline="", encoding="utf-8") as df:
            detail_writer = csv.DictWriter(df, fieldnames=detail_header)
            detail_writer.writeheader()

            # run benchmarks
            for pipeline in PIPELINES:
                print(f"\n>> Running pipeline: {pipeline}")
                for inv in inv_paths:
                    base    = os.path.basename(inv).rsplit(".", 1)[0]
                    gt_path = os.path.join(GT_DIR, f"{base}.json")
                    gt      = json.load(open(gt_path, encoding="utf-8"))

                    txt  = extract_text(inv, pipeline, use_searchable=USE_SEARCHABLE)
                    pred = ask_ollama_for_invoice_fields(txt)

                    # summary row
                    row_summary = {"invoice": base, "pipeline": pipeline}
                    correct     = 0

                    for f in fields:
                        expected = gt.get(f)
                        got      = pred.get(f)
                        ok       = is_match(expected, got)

                        # populate summary
                        row_summary[f] = 1 if ok else 0
                        if ok:
                            correct += 1

                        # write detail row
                        detail_writer.writerow({
                            "invoice":   base,
                            "pipeline":  pipeline,
                            "field":     f,
                            "expected":  expected,
                            "predicted": got,
                            "match":     1 if ok else 0,
                        })

                    row_summary["accuracy"] = correct / len(fields)
                    summary_writer.writerow(row_summary)

    print(f"\n✅ Done! Summary saved to '{OUTPUT_SUMMARY_CSV}', details to '{OUTPUT_DETAIL_CSV}'")

if __name__ == "__main__":
    main()
