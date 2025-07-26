# DATEI: utils/post_processing.py
"""
Post-Processing Modul für die Nachbearbeitung extrahierter Daten aus Rechnungen.

Dieses Modul bietet Funktionen zur Validierung und Korrektur von:
- IBAN-Nummern (mit MOD-97 Algorithmus)
- USt-IdNr. (Umsatzsteuer-Identifikationsnummern)
- Geldbeträgen (Normalisierung)
- Datumsformaten (Standardisierung)

Das Modul korrigiert nur fehlerhafte oder fehlende Eingaben und überschreibt
keine bereits korrekt extrahierten Werte.
"""
import re
from datetime import datetime
from typing import Dict, Any, List


# =============================================================================
# --- IBAN Validierung ohne externe Abhängigkeiten ---
# =============================================================================

def _char_to_num(char: str) -> str:
    """
    Konvertiert Buchstaben A-Z zu Zahlen 10-35 für IBAN-Validierung.
    
    Args:
        char: Einzelnes Zeichen (Ziffer oder Buchstabe)
        
    Returns:
        String-Repräsentation der Zahl
    """
    if char.isdigit():
        return char
    return str(ord(char.upper()) - ord('A') + 10)

def _validate_iban(iban: str) -> bool:
    """
    Validiert eine IBAN-Nummer mittels MOD-97 Algorithmus.
    
    Der MOD-97 Algorithmus ist der internationale Standard zur IBAN-Validierung:
    1. Verschiebe die ersten 4 Zeichen ans Ende
    2. Ersetze Buchstaben durch entsprechende Zahlen (A=10, B=11, ..., Z=35)
    3. Berechne den Rest der Division durch 97
    4. IBAN ist gültig, wenn Rest = 1
    
    Args:
        iban: IBAN-String (mit oder ohne Leerzeichen)
        
    Returns:
        True wenn IBAN gültig ist, False sonst
    """
    if not iban or len(iban) < 15 or len(iban) > 34:
        return False
    
    # Leerzeichen entfernen und in Großbuchstaben umwandeln
    iban = re.sub(r'\s+', '', iban.upper())
    
    # Format prüfen (2 Buchstaben + 2 Ziffern + alphanumerisch)
    if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', iban):
        return False
    
    # Erste 4 Zeichen ans Ende verschieben
    rearranged = iban[4:] + iban[:4]
    
    # Buchstaben in Zahlen umwandeln
    numeric_string = ''.join(_char_to_num(char) for char in rearranged)
    
    # MOD 97 Prüfung
    try:
        return int(numeric_string) % 97 == 1
    except ValueError:
        return False

# IBAN-Länderlängen (für zusätzliche Validierung)
# Quelle: ISO 13616 Standard
IBAN_LENGTHS = {
    'AD': 24, 'AE': 23, 'AL': 28, 'AT': 20, 'AZ': 28, 'BA': 20, 'BE': 16,
    'BG': 22, 'BH': 22, 'BR': 29, 'BY': 28, 'CH': 21, 'CR': 22, 'CY': 28,
    'CZ': 24, 'DE': 22, 'DK': 18, 'DO': 28, 'EE': 20, 'ES': 24, 'FI': 18,
    'FO': 18, 'FR': 27, 'GB': 22, 'GE': 22, 'GI': 23, 'GL': 18, 'GR': 27,
    'GT': 28, 'HR': 21, 'HU': 28, 'IE': 22, 'IL': 23, 'IS': 26, 'IT': 27,
    'JO': 30, 'KW': 30, 'KZ': 20, 'LB': 28, 'LC': 32, 'LI': 21, 'LT': 20,
    'LU': 20, 'LV': 21, 'MC': 27, 'MD': 24, 'ME': 22, 'MK': 19, 'MR': 27,
    'MT': 31, 'MU': 30, 'NL': 18, 'NO': 15, 'PK': 24, 'PL': 28, 'PS': 29,
    'PT': 25, 'QA': 29, 'RO': 24, 'RS': 22, 'SA': 24, 'SE': 24, 'SI': 19,
    'SK': 24, 'SM': 27, 'TN': 24, 'TR': 26, 'UA': 29, 'VG': 24, 'XK': 20
}


# =============================================================================
# --- Regelbasierte Verifikation und Korrektur ---
# =============================================================================

# Muster für Umsatzsteuer-Identifikationsnummern
UST_ID_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(DE[0-9]{9})\b', re.IGNORECASE),  # Deutschland
    re.compile(r'\b(ATU[0-9]{8})\b', re.IGNORECASE), # Österreich
]

# Spezifische IBAN-Muster für europäische Länder
# Jedes Muster entspricht dem exakten Format des jeweiligen Landes
IBAN_PATTERNS: List[re.Pattern] = [
    # Deutschland: DE + 2 Prüfziffern + 8 Bankleitzahl + 10 Kontonummer
    re.compile(r'\b(DE[0-9]{2}[0-9A-Z]{16})\b', re.IGNORECASE),
    # Österreich: AT + 2 Prüfziffern + 5 Bankleitzahl + 11 Kontonummer
    re.compile(r'\b(AT[0-9]{18})\b', re.IGNORECASE),
    # Schweiz: CH + 2 Prüfziffern + 5 Bankcode + 12 Kontonummer
    re.compile(r'\b(CH[0-9]{2}[0-9]{5}[0-9A-Z]{12})\b', re.IGNORECASE),
    # Niederlande: NL + 2 Prüfziffern + 4 Bankcode + 10 Kontonummer
    re.compile(r'\b(NL[0-9]{2}[A-Z]{4}[0-9]{10})\b', re.IGNORECASE),
    # Frankreich: FR + 2 Prüfziffern + 5 Bankcode + 5 Filiale + 11 Konto + 2 Schlüssel
    re.compile(r'\b(FR[0-9]{2}[0-9]{10}[0-9A-Z]{11}[0-9]{2})\b', re.IGNORECASE),
    # Belgien: BE + 2 Prüfziffern + 3 Bankcode + 7 Kontonummer + 2 Prüfziffer
    re.compile(r'\b(BE[0-9]{14})\b', re.IGNORECASE),
    # Italien: IT + 2 Prüfziffern + 1 Prüfziffer + 5 Bankcode + 5 Filiale + 12 Konto
    re.compile(r'\b(IT[0-9]{2}[A-Z][0-9]{10}[0-9A-Z]{12})\b', re.IGNORECASE),
    # Spanien: ES + 2 Prüfziffern + 4 Bankcode + 4 Filiale + 2 Prüfziffer + 10 Konto
    re.compile(r'\b(ES[0-9]{22})\b', re.IGNORECASE),
    # Vereinigtes Königreich: GB + 2 Prüfziffern + 4 Bankcode + 6 Filiale + 8 Konto
    re.compile(r'\b(GB[0-9]{2}[A-Z]{4}[0-9]{14})\b', re.IGNORECASE),
    # Weitere wichtige EU-Länder
    re.compile(r'\b(PL[0-9]{26})\b', re.IGNORECASE),  # Polen
    re.compile(r'\b(SE[0-9]{22})\b', re.IGNORECASE),  # Schweden
    re.compile(r'\b(DK[0-9]{16})\b', re.IGNORECASE),  # Dänemark
    re.compile(r'\b(NO[0-9]{13})\b', re.IGNORECASE),  # Norwegen
    re.compile(r'\b(FI[0-9]{16})\b', re.IGNORECASE),  # Finnland
]

# Fallback-Muster für andere europäische IBANs (allgemeineres Muster)
# Wird verwendet, wenn kein spezifisches Ländermuster gefunden wurde
IBAN_FALLBACK_PATTERN: re.Pattern = re.compile(r'\b([A-Z]{2}[0-9]{2}[\sA-Z0-9]{11,30})\b', re.IGNORECASE)


def verify_and_correct_fields(data: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """
    Fungiert als "Sicherheitsnetz" zur Verifikation und Korrektur von Feldern 
    mit starken Mustern. Korrigiert nur bei tatsächlichen Fehlern oder fehlenden Werten.
    
    Diese Funktion implementiert eine konservative Korrekturstrategie:
    - Prüft zuerst, ob bestehende Werte korrekt sind
    - Sucht nur bei Fehlern oder leeren Feldern nach Korrekturen im Volltext
    - Überschreibt niemals korrekte bestehende Werte
    
    Args:
        data: Dictionary mit extrahierten Daten aus dem LLM
        full_text: Vollständiger OCR-Text des Dokuments
        
    Returns:
        Korrigiertes Dictionary mit verifizierten/korrigierten Werten
        
    Korrigierte Felder:
        - iban: IBAN-Nummer (mit MOD-97 Validierung)
        - ust-id: Umsatzsteuer-Identifikationsnummer
    """
    if not isinstance(data, dict):
        return data

    # --- IBAN Verifikation und Korrektur ---
    current_iban = data.get('iban', '').strip()
    iban_needs_correction = False
    
    # Prüfen, ob aktuelle IBAN fehlt oder ungültig ist
    if not current_iban:
        iban_needs_correction = True
        print(f"[Info] Post-processing: IBAN-Feld ist leer, suche nach IBAN im Text.")
    elif not _validate_iban(current_iban):
        iban_needs_correction = True
        print(f"[Info] Post-processing: Aktuelle IBAN '{current_iban}' ist ungültig, versuche Korrektur.")
    
    if iban_needs_correction:
        iban_match = None
        candidate_iban = None

        # Zuerst versuchen wir die spezifischen europäischen IBAN-Muster
        for pattern in IBAN_PATTERNS:
            match = pattern.search(full_text)
            if match:
                candidate_iban = re.sub(r'\s+', '', match.group(1).upper())
                
                # Zusätzliche Längenvalidierung
                country_code = candidate_iban[:2]
                expected_length = IBAN_LENGTHS.get(country_code)
                if expected_length and len(candidate_iban) == expected_length:
                    # Validierung mit MOD-97 Algorithmus
                    if _validate_iban(candidate_iban):
                        iban_match = match
                        break

        # Falls kein spezifisches Muster gefunden wurde, Fallback verwenden
        if not iban_match:
            fallback_match = IBAN_FALLBACK_PATTERN.search(full_text)
            if fallback_match:
                candidate_iban = re.sub(r'\s+', '', fallback_match.group(1).upper())
                
                # Zusätzliche Längenvalidierung
                country_code = candidate_iban[:2]
                expected_length = IBAN_LENGTHS.get(country_code)
                if expected_length and len(candidate_iban) == expected_length:
                    # Validierung mit MOD-97 Algorithmus
                    if _validate_iban(candidate_iban):
                        iban_match = fallback_match

        if iban_match and candidate_iban:
            # Standardisiertes elektronisches Format (ohne Leerzeichen, Großbuchstaben)
            corrected_iban = candidate_iban
            print(f"[Info] Post-processing: IBAN korrigiert zu '{corrected_iban}'.")
            data['iban'] = corrected_iban
        else:
            # Keine gültige IBAN gefunden
            if candidate_iban:
                print(f"[Warnung] Post-processing: Ungültiger IBAN-ähnlicher String '{candidate_iban}' gefunden. Wird verworfen.")
            if not current_iban:
                print(f"[Warnung] Post-processing: Keine gültige IBAN im Text gefunden.")

    # --- USt-IdNr. Verifikation und Korrektur ---
    current_ust_id = data.get('ust-id', '').strip()
    ust_id_needs_correction = False
    
    # Prüfen, ob aktuelle USt-IdNr. fehlt oder nicht den erwarteten Mustern entspricht
    if not current_ust_id:
        ust_id_needs_correction = True
        print(f"[Info] Post-processing: USt-IdNr.-Feld ist leer, suche nach USt-IdNr. im Text.")
    else:
        # Prüfen, ob aktuelle USt-IdNr. einem gültigen Muster entspricht
        is_valid_ust_id = any(pattern.match(current_ust_id) for pattern in UST_ID_PATTERNS)
        if not is_valid_ust_id:
            ust_id_needs_correction = True
            print(f"[Info] Post-processing: Aktuelle USt-IdNr. '{current_ust_id}' entspricht nicht den erwarteten Mustern, versuche Korrektur.")
    
    if ust_id_needs_correction:
        for pattern in UST_ID_PATTERNS:
            ust_id_match = pattern.search(full_text)
            if ust_id_match:
                corrected_ust_id = ust_id_match.group(1).upper()
                print(f"[Info] Post-processing: USt-IdNr. korrigiert zu '{corrected_ust_id}'.")
                data['ust-id'] = corrected_ust_id
                break
        else:
            if not current_ust_id:
                print(f"[Warnung] Post-processing: Keine gültige USt-IdNr. im Text gefunden.")

    return data

def canon_money(x: str | float | None) -> float | None:
    """
    Normalisiert einen Geldbetrag zu einem Float-Wert.
    
    Behandelt verschiedene Eingabeformate:
    - Strings mit Währungssymbolen (€)
    - Verschiedene Dezimaltrennzeichen (, oder .)
    - Tausendertrennzeichen (Leerzeichen, Apostrophe)
    - Bereits numerische Werte
    
    Args:
        x: Geldbetrag als String, Float oder None
        
    Returns:
        Normalisierter Geldbetrag auf 2 Dezimalstellen gerundet oder None
        
    Beispiele:
        "1.234,56 €" -> 1234.56
        "1,234.56" -> 1234.56
        "1'000.00" -> 1000.00
        None -> None
    """
    if x in (None, "", "null"): 
        return None
    if isinstance(x, (int, float)): 
        return round(float(x), 2)
    
    # String-Bereinigung: Währungssymbole und Leerzeichen entfernen
    x = str(x).replace("'", "").replace(" ", "").replace("€", "").replace(",", ".")
    try:
        return round(float(x), 2)
    except ValueError:
        return None

def canon_date(date_str: str) -> str | None:
    """
    Normalisiert einen Datumsstring zum Format 'TT.MM.JJJJ'.
    
    Unterstützt verschiedene Eingabeformate:
    - TT.MM.JJJJ (bereits korrekt)
    - JJJJ-MM-TT (ISO-Format)
    - TT/MM/JJJJ
    - JJJJ/MM/TT
    
    Args:
        date_str: Datumsstring in verschiedenen Formaten
        
    Returns:
        Normalisiertes Datum im Format 'TT.MM.JJJJ' oder None bei Fehlern
        
    Beispiele:
        "2024-03-15" -> "15.03.2024"
        "15/03/2024" -> "15.03.2024"
        "15.03.2024" -> "15.03.2024"
        "ungültiges_datum" -> None
    """
    if not date_str:
        return None
    date_str = date_str.strip()
    
    # Verschiedene Datumsformate ausprobieren
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d.%m.%Y")
        except ValueError:
            continue
    return None  # Wenn kein Format passt, None zurückgeben


def finalize_extracted_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalisiert die extrahierten Rohdaten vom LLM durch Konvertierung von
    numerischen Feldern von String zu entsprechenden Zahlentypen.
    
    Diese Funktion wendet Normalisierungsregeln auf spezifische Felder an:
    - Geldbeträge werden zu Float konvertiert und gerundet
    - Daten werden zu einheitlichem Format normalisiert
    
    Args:
        data: Rohdictionary mit String-Werten vom LLM
        
    Returns:
        Finalisiertes Dictionary mit normalisierten Datentypen
        
    Behandelte Felder:
        - total_amount: Gesamtbetrag (Float)
        - tax_rate: Steuersatz (Float) 
        - invoice_date: Rechnungsdatum (String im Format TT.MM.JJJJ)
    """
    if not isinstance(data, dict):
        return data

    # Gesamtbetrag normalisieren
    if 'total_amount' in data:
        data['total_amount'] = canon_money(data['total_amount'])

    # Steuersatz normalisieren
    if 'tax_rate' in data:
        data['tax_rate'] = canon_money(data['tax_rate'])

    # Rechnungsdatum normalisieren
    if 'invoice_date' in data:
        data['invoice_date'] = canon_date(data['invoice_date'])

    return data