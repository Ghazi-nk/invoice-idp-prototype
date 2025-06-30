#!/usr/bin/env python3
"""
This script discovers and tests all invoice-processing modules by comparing their outputs
against expected results, and generates a timestamped report in app/results/reports.
"""
import os
import json
import importlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def load_expected(expected_file: str) -> Optional[Dict[str, Any]]:
    """
    Load the expected output JSON for a given invoice test.
    """
    if not os.path.exists(expected_file):
        print(f"[Warning] Expected file not found: {expected_file}")
        return None
    with open(expected_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_dicts(actual: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Compare two dictionaries and return whether they match and a list of differences.
    """
    diffs: List[str] = []
    keys = set(actual.keys()) | set(expected.keys())
    for key in keys:
        a = actual.get(key)
        e = expected.get(key)
        if a != e:
            diffs.append(f"Field '{key}': expected={e!r}, got={a!r}")
    return (not diffs, diffs)


def run_tests_for_module(
    module_name: str,
    pipeline_module: str,
    func_name: str,
    test_dir: str,
    expected_dir: str
) -> Dict[str, Any]:
    """
    Run all PDF tests for a single pipeline module.
    Returns a dict with summary and detailed results.
    """
    print(f"\n[Module] Testing '{module_name}'...")
    try:
        mod = importlib.import_module(pipeline_module)
    except ImportError as e:
        return {"error": f"ImportError: {e}"}

    if not hasattr(mod, func_name):
        return {"error": f"Function '{func_name}' not found in {pipeline_module}"}

    func = getattr(mod, func_name)
    try:
        pdfs = [f for f in os.listdir(test_dir) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        return {"error": f"Test directory not found: {test_dir}"}

    results = []
    for pdf in pdfs:
        pdf_path = os.path.join(test_dir, pdf)
        base = os.path.splitext(pdf)[0]
        expected_file = os.path.join(expected_dir, base + '.json')

        print(f"[Info] Processing {pdf}...")
        actual = func(pdf_path, True)
        expected = load_expected(expected_file)

        if actual is None or expected is None:
            status = 'error'
            diffs = [f"Missing actual or expected data: {actual} vs {expected_file}"]
        else:
            match, diffs = compare_dicts(actual, expected)
            status = 'pass' if match else 'fail'

        results.append({
            'invoice': pdf,
            'status': status,
            'differences': diffs
        })

    summary = {
        'total': len(results),
        'passed': sum(1 for r in results if r['status']=='pass'),
        'failed': sum(1 for r in results if r['status']=='fail'),
        'errored': sum(1 for r in results if r['status']=='error')
    }

    return {'summary': summary, 'details': results}


def generate_report(all_results: Dict[str, Any], report_path: str) -> None:
    """
    Write the aggregated test report to disk.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(report_path, 'w', encoding='utf-8') as rpt:
        rpt.write(f"Test Report - {now}\n")
        rpt.write("="*50 + "\n\n")
        for module, data in all_results.items():
            rpt.write(f"Module: {module}\n")
            if 'error' in data:
                rpt.write(f"Error: {data['error']}\n\n")
                continue
            summ = data['summary']
            rpt.write(f"Total: {summ['total']}, Passed: {summ['passed']}, ")
            rpt.write(f"Failed: {summ['failed']}, Errored: {summ['errored']}\n")
            rpt.write("-"*30 + "\n")
            for d in data['details']:
                if d['status'] != 'pass':
                    rpt.write(f"{d['invoice']}: {d['status']}\n")
                    for diff in d['differences']:
                        rpt.write(f"  * {diff}\n")
            rpt.write("\n")
    print(f"[Info] Report written to {report_path}")


def main():
    # Base directories
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    resources = os.path.join(base, 'resources')
    report_dir = os.path.join(base, 'app', 'results', 'reports')
    os.makedirs(report_dir, exist_ok=True)

    # Handle possible folder name typo for invoices
    invoices_dir = os.path.join(resources, 'Invoices')
    if not os.path.isdir(invoices_dir):
        alt = os.path.join(resources, 'Inovices')  # alternate spelling
        if os.path.isdir(alt):
            invoices_dir = alt
        else:
            print(f"[Error] Neither 'Invoices' nor 'Inovices' folder found in {resources}")
            return

    expected_dir = os.path.join(resources, 'GroundTruth')
    if not os.path.isdir(expected_dir):
        print(f"[Error] 'GroundTruth' folder not found in {resources}")
        return

    # Module configurations
    modules = [
        {
            'name': 'OCR Ollama',
            'pipeline_module': 'app.ocr_ollama.pipeline',
            'function': 'process_single_pdf',
            'test_dir': invoices_dir,
            'expected_dir': expected_dir
        },
        # add more modules here
    ]

    results = {}
    for m in modules:
        results[m['name']] = run_tests_for_module(
            m['name'], m['pipeline_module'], m['function'],
            m['test_dir'], m['expected_dir']
        )

    # Filename: dd_mm_yy.txt
    fname = datetime.now().strftime('%d_%m_%y') + '.txt'
    path = os.path.join(report_dir, fname)
    generate_report(results, path)

if __name__ == '__main__':
    main()

