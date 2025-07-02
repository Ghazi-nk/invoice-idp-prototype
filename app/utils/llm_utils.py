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

def ask_ollama_for_invoice_fields(ocr_text: str) -> dict:
    prompt = f"{PROMPT_TXT}\n{ocr_text}\n"
    body   = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    requests.packages.urllib3.disable_warnings()
    resp = requests.post(GENERATE_ENDPOINT, json=body, verify=False, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API: {resp.status_code} – {resp.text}")

    raw = resp.json().get("response", "")

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

# if __name__ == "__main__":
#     txt = """
#                                                                                                   MM
#                                                                                                  MAX    MUSTERMANN
#                Abs.:Mustermann  GmbH.  HauptstraBe 123  5020Salzburg
#                  ABC   GmbH
#                  Herr' Wolfram   Svenske
#                  BoschstraBe   34
#                  44787   BOCHUM
#                  DEUTSCHLAND
#                                                                                          Seite:
#                                                                                                                                    1
#                                                                                          Kunden  Nr.:
#                                                                                                                                   16
#                                                                                          Datum:
#                                                                                                                           13.10.2020
#                Rechnung         Re-9/2020
#                vielen Dank  fur die Beauftragung.
#                Pos
#                              Menge
#                                              Text
#                                                                                                      Einzelpreis
#                                                                                                                        Gesamtpreis
#                                                                                                          EUR
#                                                                                                                            EUR
#                   1
#                                 Std.
#                                              Arbeitsstunde  Meister
#                                                                                                               74,99
#                                                                                                                                  62,49
#                Gesamt  Netto
#                                                                                                                                   62,49
#                zzgl.19,00 % USt. auf
#                                                                                                               62,49
#                                                                                                                                   12,50
#                Gesamtbertrag
#                                                                                                                                  74,99
#                Zahlungsziel:  14 Tage
#                Bitte uberweisen   Sie den  Rechnungsbetrag     auf unser  Konto:
#                IBAN: AT491700000122001632        BIC:  BFKKDE2K    Inhaber: Mustermann     GmbH.
#                Mit freundlichen   GruBen,
#                Max  Mustermann
#                                                                   Mustermann GmbH.
#                                                         HauptstraBe123 5020 SalzburgOsterreich
#                                                       Tel.:+41234 123456 ofice@mustermann.com
#                                           St.-Nr:23/123/456/789UID:ATU12345678  IBAN:AT491700000122001632
# """
#     extracted = ask_ollama_for_invoice_fields(txt)
#     print(extracted)
#     print(json.dumps(extracted, indent=2, ensure_ascii=False))

