"""
Benchmark-System für die Evaluation von IDP-Pipeline-Performance.

Dieses Modul implementiert ein umfassendes Benchmark-System zur systematischen
Evaluation verschiedener OCR-Engines und Pipeline-Konfigurationen. Es führt
automatisierte Tests gegen Ground-Truth-Daten durch und generiert detaillierte
Performance-Metriken.

Funktionen:
- Multi-Engine OCR-Benchmarking (Tesseract, PaddleOCR, EasyOCR, DocTR, LayoutLMv3)
- Ground-Truth-Vergleich mit Precision, Recall, F1-Score
- Parallelisierte Verarbeitung für bessere Performance
- CSV-Export für wissenschaftliche Auswertung
- Aggregierte Pipeline-Performance-Analyse

Metriken:
- Accuracy: Prozentuale Übereinstimmung der Felder
- Precision/Recall/F1: Klassische ML-Evaluationsmetriken
- Success Rate: Anteil erfolgreich verarbeiteter Dokumente
- Processing Duration: Zeitanalyse für Performance-Optimierung

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import json
import csv
import time
import os
from pathlib import Path
import multiprocessing
import collections
import statistics
import sys
import fitz


from app.pipeline import extract_invoice_fields_from_pdf
from app.ocr.ocr_manager import get_available_engines
from app.ocr.pdf_utils import extract_text_if_searchable
from app.semantic_extraction import ollama_extract_invoice_fields

from app.post_processing import finalize_extracted_fields, verify_and_correct_fields

from app.benchmark.evaluation_utils import is_match, check_success


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Ermittelt die Anzahl der Seiten in einer PDF-Datei.
    
    Args:
        pdf_path (str): Pfad zur PDF-Datei
        
    Returns:
        int: Anzahl der Seiten oder 0 bei Fehlern
        
    Note:
        Verwendet PyMuPDF für robuste PDF-Verarbeitung.
    """
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        return page_count
    except Exception as e:
        print(f"Error getting page count for {pdf_path}: {e}")
        return 0

# --- Configuration for benchmark ---
from app.config import (
    INVOICES_DIR,
    LABELS_DIR,
    OLLAMA_MODEL
)

# --- Ensure required config variables are set ---
if INVOICES_DIR is None:
    print("Error: INVOICES_DIR is not set.", file=sys.stderr)
    sys.exit(1)
if LABELS_DIR is None:
    print("Error: LABELS_DIR is not set.", file=sys.stderr)
    sys.exit(1)
if OLLAMA_MODEL is None:
    print("Error: OLLAMA_MODEL is not set.", file=sys.stderr)
    sys.exit(1)

NUM_WORKERS = 2
MODEL_SAFE = OLLAMA_MODEL.replace(":", "_")
BENCHMARK_DIR = Path(__file__).parent
OUTPUT_SUMMARY_CSV = str(BENCHMARK_DIR / f"summary_{MODEL_SAFE}.csv")
OUTPUT_DETAIL_CSV = str(BENCHMARK_DIR / f"details_{MODEL_SAFE}.csv")
OUTPUT_RESULTS_CSV = str(BENCHMARK_DIR / f"results_{MODEL_SAFE}.csv")


def calc_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Berechnet Precision, Recall und F1-Score aus Konfusionsmatrix-Werten.
    
    Diese Funktion implementiert die klassischen ML-Evaluationsmetriken
    für binäre Klassifikation basierend auf True Positives, False Positives
    und False Negatives.
    
    Args:
        tp (int): True Positives (korrekt erkannte Felder)
        fp (int): False Positives (fälschlich erkannte Felder) 
        fn (int): False Negatives (verpasste Felder)
        
    Returns:
        tuple[float, float, float]: (Precision, Recall, F1-Score) gerundet auf 3 Stellen

    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def process_task(task_args: tuple, is_searchable: bool) -> dict | None:
    """
    Processes a single (pdf_path, engine, label_path) task.
    
    Args:
        task_args: Tuple of (pdf_path, engine, label_path)
        is_searchable: Whether the PDF is searchable
    """
    pdf_path, engine, label_path = task_args
    base_name = pdf_path.stem

    engine_name = engine
    
    print(f"  Processing '{base_name}.pdf' with '{engine_name}' (searchable={is_searchable})...")
    try:
        start_time = time.perf_counter()
        with open(label_path, encoding="utf-8") as f:
            gt = json.load(f)

        if engine == "searchable":
            page_texts = extract_text_if_searchable(str(pdf_path))
            # Clean text if needed

            llm_output, ollama_duration = ollama_extract_invoice_fields(page_texts)
            full_text_for_verification = "\n".join(page_texts)
            pred = finalize_extracted_fields(verify_and_correct_fields(llm_output, full_text_for_verification))
            total_duration = time.perf_counter() - start_time
            processing_duration = total_duration - ollama_duration
        else:
            pred, ollama_duration, processing_duration = extract_invoice_fields_from_pdf(
                pdf_path=str(pdf_path),
                engine=engine
            )
            total_duration = ollama_duration + processing_duration
        
        # Round durations to 3 decimal places
        ollama_duration = round(ollama_duration, 3)
        processing_duration = round(processing_duration, 3)
        total_duration = round(total_duration, 3)

        # Get page count
        page_count = get_pdf_page_count(str(pdf_path))
        
        fields = list(gt.keys())
        summary_row = {"invoice": base_name, "pipeline": engine_name, 
                      "ollama_duration": ollama_duration, 
                      "processing_duration": processing_duration, 
                      "total_duration": total_duration, 
                      "page_count": page_count,
                      "searchable": is_searchable}
        detail_rows = []
        correct = tp = fp = fn = 0

        for fld in fields:
            match = is_match(fld, gt.get(fld), pred.get(fld))
            summary_row[fld] = int(match)
            if match:
                correct += 1
                if gt.get(fld) not in (None, "", "null"): tp += 1
            else:
                gt_has = gt.get(fld) not in (None, "", "null")
                pred_has = pred.get(fld) not in (None, "", "null")
                if gt_has: fn += 1
                if pred_has: fp += 1

            detail_rows.append({
                "invoice": base_name, "pipeline": engine_name, "field": fld,
                "expected": gt.get(fld), "predicted": pred.get(fld), "match": int(match), 
                "searchable": is_searchable
            })

        summary_row["accuracy"] = round(correct / len(fields), 3)
        prec, rec, f1 = calc_metrics(tp, fp, fn)
        summary_row.update(precision=prec, recall=rec, f1=f1)

        # --- NEW: Check business rule success ---
        summary_row["success"] = int(check_success(gt, pred))
        summary_row["success"] = int(check_success(gt, pred))

        print(f"  ✓ Finished '{base_name}.pdf' with '{engine_name}' in {total_duration}s.")
        return {"summary": summary_row, "details": detail_rows}

    except Exception as e:
        print(f"  ✗ ERROR processing '{base_name}.pdf' with '{engine_name}': {e}")
        return None


def process_task_with_args(args):
    return process_task(*args)


def generate_final_results():
    """Calculates and saves the aggregated results for each pipeline."""
    if not os.path.exists(OUTPUT_SUMMARY_CSV):
        print("Warning: summary.csv not found. Cannot generate final results.")
        return

    pipeline_metrics = collections.defaultdict(lambda: {
        'accuracies': [], 'precisions': [], 'recalls': [], 'f1s': [], 
        'ollama_durations': [], 'processing_durations': [], 'total_durations': [], 
        'successes': []
    })

    with open(OUTPUT_SUMMARY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pipeline = row['pipeline']
            pipeline_metrics[pipeline]['accuracies'].append(float(row['accuracy']))
            pipeline_metrics[pipeline]['precisions'].append(float(row['precision']))
            pipeline_metrics[pipeline]['recalls'].append(float(row['recall']))
            pipeline_metrics[pipeline]['f1s'].append(float(row['f1']))
            
            # Check if we have the new duration fields or the old 'duration' field
            if 'total_duration' in row:
                pipeline_metrics[pipeline]['total_durations'].append(float(row['total_duration']))
                pipeline_metrics[pipeline]['ollama_durations'].append(float(row['ollama_duration']))
                pipeline_metrics[pipeline]['processing_durations'].append(float(row['processing_duration']))
            elif 'duration' in row:  # Backward compatibility
                pipeline_metrics[pipeline]['total_durations'].append(float(row['duration']))
                # Set default values for the new duration fields
                pipeline_metrics[pipeline]['ollama_durations'].append(0.0)
                pipeline_metrics[pipeline]['processing_durations'].append(float(row['duration']))
            
            # Handle both old 'acceptance' and new 'success' columns for backward compatibility
            if 'success' in row:
                pipeline_metrics[pipeline]['successes'].append(float(row['success']))
            elif 'acceptance' in row:
                pipeline_metrics[pipeline]['successes'].append(float(row['acceptance']))

    header = ["pipeline", "mean_accuracy", "mean_precision", "mean_recall", "mean_f1", 
              "mean_ollama_duration", "mean_processing_duration", "mean_total_duration",
              "success_rate"]
    with open(OUTPUT_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for pipeline, data in sorted(pipeline_metrics.items()):
            writer.writerow({
                "pipeline": pipeline,
                "mean_accuracy": round(statistics.mean(data['accuracies']), 3),
                "mean_precision": round(statistics.mean(data['precisions']), 3),
                "mean_recall": round(statistics.mean(data['recalls']), 3),
                "mean_f1": round(statistics.mean(data['f1s']), 3),
                "mean_ollama_duration": round(statistics.mean(data['ollama_durations']), 3),
                "mean_processing_duration": round(statistics.mean(data['processing_durations']), 3),
                "mean_total_duration": round(statistics.mean(data['total_durations']), 3),
                "success_rate": round(statistics.mean(data['successes']), 3)
            })


def main():
    """
    Hauptfunktion zur Ausführung des Benchmark-Systems.
    
    Koordiniert die gesamte Benchmark-Ausführung: Datei-Discovery,
    Parallelisierung, Engine-Testing und Ergebnis-Aggregation.
    Erstellt umfassende CSV-Reports für wissenschaftliche Auswertung.
    
    Workflow:
        1. Invoice- und Label-Dateien scannen
        2. Durchsuchbare PDFs identifizieren  
        3. Task-Matrix erstellen (PDFs × Engines)
        4. Parallelisierte Verarbeitung mit multiprocessing
        5. Ergebnis-Sammlung und CSV-Export
        6. Aggregierte Statistiken generieren
        
    Output-Dateien:
        - details_{model}.csv: Detaillierte Feld-Evaluationen
        - summary_{model}.csv: Zusammenfassung pro Rechnung
        - results_{model}.csv: Aggregierte Pipeline-Performance
    """
    completed_work = set()
    if os.path.exists(OUTPUT_SUMMARY_CSV):
        with open(OUTPUT_SUMMARY_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row: completed_work.add((row[0], row[1]))
    print(f"Found {len(completed_work)} previously completed tasks.")

    tasks_to_do = []
    all_pdfs = sorted(Path(INVOICES_DIR).glob("*.pdf"))
    all_engines = get_available_engines()
    all_engines_with_searchable = all_engines + ["searchable"]

    pdf_searchable_map = {}
    for pdf_path in all_pdfs:
        base_name = pdf_path.stem
        label_path = Path(LABELS_DIR) / f"{base_name}.json"
        if not label_path.exists():
            print(f"  ! Warning: No label file for '{base_name}.pdf'. Skipping.")
            continue
        # Determine if PDF is searchable
        try:
            text_pages = extract_text_if_searchable(str(pdf_path))
            is_searchable = any(page.strip() for page in text_pages)
        except Exception as e:
            print(f"  ! Error checking if '{base_name}.pdf' is searchable: {e}")
            is_searchable = False
        pdf_searchable_map[base_name] = is_searchable
        
        for engine in all_engines:
            # Add task without bbox
            regular_engine_name = engine
            if (base_name, regular_engine_name) not in completed_work:
                tasks_to_do.append(((pdf_path, engine, label_path), is_searchable))
                
        # Only add 'searchable' engine if PDF is searchable
        if is_searchable and (base_name, "searchable") not in completed_work:
            tasks_to_do.append(((pdf_path, "searchable", label_path), True))

    if not tasks_to_do:
        print("✓ No new tasks to process. Benchmark is up-to-date.")
    else:
        print(f"Created {len(tasks_to_do)} new tasks to process.")
        summary_exists = os.path.exists(OUTPUT_SUMMARY_CSV)
        details_exists = os.path.exists(OUTPUT_DETAIL_CSV)

        with open(OUTPUT_SUMMARY_CSV, "a" if summary_exists else "w", newline="", encoding="utf-8", buffering=1) as sum_f, \
                open(OUTPUT_DETAIL_CSV, "a" if details_exists else "w", newline="", encoding="utf-8", buffering=1) as det_f:

            first_label_path = Path(LABELS_DIR) / f"{all_pdfs[0].stem}.json"
            with open(first_label_path, encoding="utf-8") as fh:
                fields = list(json.load(fh).keys())

            # --- Update header to include the new duration fields ---
            header_summary = ["invoice", "pipeline", *fields, "accuracy", "precision", 
                             "recall", "f1", "success", "ollama_duration", 
                             "processing_duration", "total_duration", "page_count", "searchable"]
            header_details = ["invoice", "pipeline", "field", "expected", "predicted", 
                             "match", "searchable"]

            summary_writer = csv.DictWriter(sum_f, fieldnames=header_summary)
            detail_writer = csv.DictWriter(det_f, fieldnames=header_details)

            if not summary_exists: summary_writer.writeheader()
            if not details_exists: detail_writer.writeheader()

            if NUM_WORKERS > 1:
                print(f"\n▶ Running in PARALLEL mode with {NUM_WORKERS} workers...")
                with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                    for result in pool.imap_unordered(process_task_with_args, tasks_to_do):
                        if result:
                            summary_writer.writerow(result["summary"])
                            detail_writer.writerows(result["details"])
            else:
                print("\n▶ Running in SEQUENTIAL mode (invoice-centric)...")
                tasks_by_pdf = collections.defaultdict(list)
                for task in tasks_to_do:
                    tasks_by_pdf[task[0][0]].append(task)
                for pdf_path, pdf_tasks in sorted(tasks_by_pdf.items()):
                    print(f"\n-- Processing PDF: {pdf_path.name} --")
                    for task in pdf_tasks:
                        result = process_task(*task)
                        if result:
                            summary_writer.writerow(result["summary"])
                            detail_writer.writerows(result["details"])

    print(f"\n✓ Benchmark complete.")
    generate_final_results()
    print(f"✓ Final results summary generated at {OUTPUT_RESULTS_CSV}")


if __name__ == "__main__":
    main()