import json
import os
import sys

import requests
from pathlib import Path
from utils.config import GENERATE_ENDPOINT, OLLAMA_MODEL, TMP_DIR


import json
import re                #  <-- neu
import requests
from pathlib import Path
from utils.config import GENERATE_ENDPOINT, OLLAMA_MODEL

# ... (BASE_DIR, PROMPT_FILE etc. wie gehabt) ...
BASE_DIR    = Path(__file__).resolve().parents[2]
PROMPT_FILE = BASE_DIR / "resources" / "Prompts" / "prompt.txt"
PROMPT_TXT  = PROMPT_FILE.read_text(encoding="utf-8")

def ollama_extract_invoice_fields(ocr_text: str) -> dict:
    prompt = f"{PROMPT_TXT}\n{ocr_text}\n"
    body   = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    requests.packages.urllib3.disable_warnings()
    resp = requests.post(GENERATE_ENDPOINT, json=body, verify=False, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API: {resp.status_code} – {resp.text}")

    raw = resp.json().get("response", "")
    print(f"Ollama response: {raw}")
    json_chunk = _extract_first_complete_json(raw)
    if not json_chunk:
        raise ValueError("Could not find a complete JSON object in Ollama response:\n" + raw[:500])

    # ─── PATCH: äußere Doppelklammern {{ … }} entfernen ──────────────────────────
    # akzeptiert exakt zwei geschweifte Klammern am Anfang und Ende
    json_chunk = re.sub(r'^\{\{(.*)\}\}$', r'{\1}', json_chunk, count=1, flags=re.DOTALL)
    # ------------------------------------------------------------------------------

    try:
        return json.loads(json_chunk)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON still malformed after brace balancing: {e}\nChunk:\n{json_chunk!r}")


def _extract_first_complete_json(text: str) -> str | None:
    """
    Return the first brace-balanced JSON object found in *text*,
    or None if no complete object exists.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth, in_string, escape = 0, False, False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return None

if __name__ == "__main__":
    txt = """   
    {"text": "USTERMANN", "bbox": [727, 71, 814, 80]}
    {"text": "Rechnung", "bbox": [587, 110, 679, 122]}
    {"text": "Mustermann Malerei GmbH.,", "bbox": [585, 126, 820, 137]}
    {"text": "Abs: Mustermann GmbH e Hauptstrasse 123 eÂ¢ 13507 Berlin", "bbox": [105, 140, 507, 149]}
    {"text": "Hauptstrasse 123, 13507 Berlin", "bbox": [585, 140, 846, 152]}
    {"text": "+49 622 87654321", "bbox": [652, 155, 811, 164]}
    {"text": "Best Pinsel . KG", "bbox": [120, 163, 240, 173]}
    {"text": "Tel/Fax", "bbox": [583, 155, 644, 165]}
    {"text": "mustermann@mustermann.com", "bbox": [585, 169, 850, 180]}
    {"text": "Frau Mag. Veronika Huber", "bbox": [120, 180, 340, 192]}
    {"text": "http: //www.mustermann.com", "bbox": [585, 183, 831, 194]}
    {"text": "Herman Leopoldi Gasse 23", "bbox": [120, 201, 344, 212]}
    {"text": "1220 Wien", "bbox": [121, 217, 208, 227]}
    {"text": "Osterreich", "bbox": [120, 232, 204, 243]}
    {"text": "Datum: 06.11.2012", "bbox": [730, 274, 895, 283]}
    {"text": "Rechnung: Re-21/2012", "bbox": [700, 290, 895, 301]}
    {"text": "Rechnungsvorlage mit Bild", "bbox": [107, 341, 357, 352]}
    {"text": "Sehr geehrte Frau Mag. Huber,", "bbox": [107, 362, 367, 374]}
    {"text": "vielen Dank fiir Ihr Vertrauen in unsere Produkte. Wir hoffen Sie sind zufrieden und wurden", "bbox": [109, 382, 872, 391]}
    {"text": "uns freuen wieder von Ihnen zu horen.", "bbox": [110, 396, 431, 405]}
    {"text": "ist ein", "bbox": [158, 425, 208, 434]}
    {"text": "der", "bbox": [280, 425, 307, 434]}
    {"text": "mit", "bbox": [490, 425, 520, 434]}
    {"text": "(Dies Bespiel Rechnungsvorlage Bild.)", "bbox": [110, 425, 574, 437]}
    {"text": "Pos. Bezeichnung", "bbox": [107, 463, 287, 474]}
    {"text": "USt. Menge Preis Gesamt", "bbox": [499, 461, 902, 476]}
    {"text": "1 Farbe blau - Eimer 2,5 Liter", "bbox": [108, 489, 399, 499]}
    {"text": "19% 10,00 Eimer 75,63 âĤ¬ 756,30 âĤ¬", "bbox": [502, 484, 903, 499]}
    {"text": "1 Eimer mit 2,5 Liter Inhalt.", "bbox": [168, 504, 397, 513]}
    {"text": "Farbe", "bbox": [220, 503, 260, 517]}
    {"text": "9", "bbox": [177, 597, 291, 627]}
    {"text": "2 Kleinmaterial", "bbox": [107, 631, 279, 643]}
    {"text": "19%", "bbox": [502, 632, 534, 641]}
    {"text": "16,80 âĤ¬", "bbox": [711, 632, 768, 642]}
    {"text": "16,80 âĤ¬", "bbox": [844, 632, 900, 642]}
    {"text": "Netto:", "bbox": [683, 669, 732, 682]}
    {"text": "649,67 âĤ¬", "bbox": [826, 670, 900, 681]}
    {"text": "19% USt:", "bbox": [655, 687, 732, 696]}
    {"text": "123,43 âĤ¬", "bbox": [827, 687, 900, 698]}
    {"text": "Brutto:", "bbox": [670, 703, 733, 712]}
    {"text": "773,10 âĤ¬", "bbox": [818, 703, 900, 714]}
    {"text": "Leistungsdatum: 05.11.2012", "bbox": [107, 726, 350, 738]}
    {"text": "Zahlungsziel: prompt", "bbox": [110, 748, 285, 760]}
    {"text": "Inhaber: Max Mustermann", "bbox": [110, 776, 371, 789]}
    {"text": "Konto-Nr.: 123456789", "bbox": [473, 776, 686, 789]}
    {"text": "Bank:", "bbox": [110, 790, 155, 803]}
    {"text": "Meine Bank", "bbox": [231, 791, 326, 800]}
    {"text": "BIC:", "bbox": [473, 791, 506, 800]}
    {"text": "xswqawdad", "bbox": [593, 791, 687, 802]}
    {"text": "12345", "bbox": [232, 805, 281, 814]}
    {"text": "IBAN:", "bbox": [473, 805, 518, 814]}
    {"text": "1234567890", "bbox": [595, 805, 696, 814]}
    {"text": "BLZ:", "bbox": [110, 804, 145, 817]}
    {"text": "Mit freundlichen GruBen", "bbox": [110, 848, 307, 857]}
    {"text": "Max Mustermann", "bbox": [110, 891, 250, 900]}
    {"text": "Mustermann Malerei GmbH. Â« Hauptstrasse 123 Â¢ 13507 Berlin", "bbox": [289, 918, 713, 927]}
    {"text": "UID-Nummer: DE12345678", "bbox": [409, 929, 593, 937]}
    """
    extracted = ollama_extract_invoice_fields(txt)
    print(extracted)
    print(json.dumps(extracted, indent=2, ensure_ascii=False))

