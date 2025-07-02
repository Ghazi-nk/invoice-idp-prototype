import os
import json
import glob
import csv
import re
import string
import unicodedata
from pathlib import Path

from utils.config import INVOICES_DIR, GROUND_TRUTH_DIR as GT_DIR, OLLAMA_MODEL
from utils.pdf_utils import pdf_to_png
from ocr import tesseract_png_to_text, easyocr_png_to_text
from layoutlmv3_png2txt import layoutlm_image_to_text
from doctr_pdf2txt import doctr_pdf_to_text
from utils.searchable_pdf import extract_text_if_searchable
from utils.llm_utils import ask_ollama_for_invoice_fields

# ──────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_SAFE         = OLLAMA_MODEL.replace(":", "_")
OUTPUT_SUMMARY_CSV = f"benchmark_summary_{MODEL_SAFE}.csv"
OUTPUT_DETAIL_CSV  = f"benchmark_details_{MODEL_SAFE}.csv"
USE_SEARCHABLE     = False                     # eingebetteten Text bevorzugen
PIPELINES          = ["easyocr", "tesseract", "layoutlm", "doctr"]

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Canonicalisation & Normalisation
# ──────────────────────────────────────────────────────────────────────────────

_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


def canon_text(s: str | None) -> str:
    """Lower‑case, de‑accent, trim trailing punctuation & collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.strip().rstrip(".:,")
    s = s.lower()
    s = ''.join(c for c in s if c.isalnum() or c.isspace())
    return ' '.join(s.split())


def canon_money(x: str | float | None) -> float | None:
    if x in (None, "", "null"):
        return None
    if isinstance(x, (int, float)):
        return round(float(x), 2)
    x = str(x).replace("'", "").replace(" ", "").replace("€", "").replace(",", ".")
    try:
        return round(float(x), 2)
    except ValueError:
        return None


def canon_id(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


# ──────────────────────────────────────────────────────────────────────────────
# Vergleichs‑Logik
# ──────────────────────────────────────────────────────────────────────────────

ID_KEYS     = {"ust-id", "tax_id", "iban"}
MONEY_KEYS  = {"gross_amount", "total_amount"}


def is_match(field: str, true_val, pred_val) -> bool:
    """Vergleicht Ground‑Truth und Prediction feldspezifisch robust."""

    # Treat empty/None uniformly
    if true_val in (None, "", "null"):
        return pred_val in (None, "", "null")

    if field in MONEY_KEYS:
        return canon_money(true_val) == canon_money(pred_val)

    if field in ID_KEYS:
        return canon_id(true_val) == canon_id(pred_val)

    # fallback: plain text
    return canon_text(true_val) == canon_text(pred_val)


# ──────────────────────────────────────────────────────────────────────────────
# OCR‑Clean‑Up (vor dem Prompt)
# ──────────────────────────────────────────────────────────────────────────────

def clean_ocr_text(txt: str) -> str:
    """Heuristiken gegen häufige OCR‑Artefakte."""
    if not txt:
        return ""
    # 1. A7 → AT  (IBAN‑Vertipper)
    txt = re.sub(r"\bA7(\d{2})", r"AT\1", txt)
    # 2. Apostroph‑Tausender entfernen 1'031.10 → 1031.10
    txt = txt.replace("'", "")
    # 3. Encoding‑Müll (Â€ etc.)
    txt = txt.replace("Â€", "€")
    # 4. vereinheitliche Dezimaltrennzeichen
    txt = re.sub(r"(\d),(\d{2})(\s*€?)", r"\1.\2\3", txt)
    # 5. Silbentrennungen Wort-\nFortsetzung → WortFortsetzung
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    return txt


# ──────────────────────────────────────────────────────────────────────────────
# Text‑Extraktion pro Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(pdf_path: str, pipeline: str, use_searchable: bool = False) -> str | None:
    if use_searchable:
        txt = extract_text_if_searchable(pdf_path)
        if txt:
            return clean_ocr_text(txt)

    if pipeline == "doctr":
        txt = doctr_pdf_to_text(pdf_path)
    elif pipeline == "easyocr":
        png = pdf_to_png(pdf_path)
        txt = easyocr_png_to_text(png)
    elif pipeline == "tesseract":
        png = pdf_to_png(pdf_path)
        txt = tesseract_png_to_text(png)
    elif pipeline == "layoutlm":
        png = pdf_to_png(pdf_path)
        txt = layoutlm_image_to_text(png)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    return clean_ocr_text(txt)


# ──────────────────────────────────────────────────────────────────────────────
# Main Benchmark Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    inv_paths: list[str] = sorted(Path(INVOICES_DIR).glob("*.pdf"))
    if not inv_paths:
        raise RuntimeError(f"No PDFs found in {INVOICES_DIR}")

    # Feldliste aus erstem Ground‑Truth‑JSON bestimmen
    first_base = inv_paths[0].stem
    with open(Path(GT_DIR)/f"{first_base}.json", encoding="utf-8") as fh:
        fields = list(json.load(fh).keys())

    # Summary + Detail CSV vorbereiten
    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as sum_f, \
         open(OUTPUT_DETAIL_CSV,  "w", newline="", encoding="utf-8") as det_f:

        summary_writer = csv.DictWriter(sum_f, fieldnames=["invoice", "pipeline", *fields, "accuracy"])
        detail_writer  = csv.DictWriter(det_f, fieldnames=["invoice", "pipeline", "field", "expected", "predicted", "match"])
        summary_writer.writeheader()
        detail_writer.writeheader()

        # Pipelines durchlaufen
        for pipeline in PIPELINES:
            print(f"\n▶ Running pipeline: {pipeline}")
            for pdf_path in inv_paths:
                base = pdf_path.stem
                gt   = json.load(open(Path(GT_DIR)/f"{base}.json", encoding="utf-8"))

                ocr_txt = extract_text(str(pdf_path), pipeline, use_searchable=USE_SEARCHABLE)
                pred    = ask_ollama_for_invoice_fields(ocr_txt)

                # Summary‑Row aufbauen
                row_sum = {"invoice": base, "pipeline": pipeline}
                correct = 0
                for fld in fields:
                    ok = is_match(fld, gt.get(fld), pred.get(fld))
                    row_sum[fld] = int(ok)
                    correct += ok

                    # Detail‑CSV
                    detail_writer.writerow({
                        "invoice":   base,
                        "pipeline":  pipeline,
                        "field":     fld,
                        "expected":  gt.get(fld),
                        "predicted": pred.get(fld),
                        "match":     int(ok)
                    })

                row_sum["accuracy"] = round(correct / len(fields), 3)
                summary_writer.writerow(row_sum)

    print(f"\n✓ Done – wrote {OUTPUT_SUMMARY_CSV} & {OUTPUT_DETAIL_CSV}")


if __name__ == "__main__":
    main()
