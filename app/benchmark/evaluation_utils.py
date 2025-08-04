"""
Evaluation-Utilities für Benchmark-Vergleiche und Metriken.

Dieses Modul stellt spezialisierte Funktionen für den Vergleich von
extrahierten Daten mit Ground-Truth-Referenzen bereit. Es implementiert
robuste Vergleichsalgorithmen, die typische Variationen in OCR-Ergebnissen
und LLM-Ausgaben berücksichtigen.

Funktionen:
- Textnormalisierung für robuste Vergleiche
- Spezialisierte Vergleichsfunktionen für Namen, Zahlen, Daten
- Success-Rate-Berechnung basierend auf Feldabdeckung

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import re
import unicodedata
from typing import Dict

from app.post_processing import canon_number


def canon_text(s: str | None) -> str:
    """
    Kanonisiert einen Text für robuste Vergleiche.
    
    Diese Funktion normalisiert Texte durch Unicode-Normalisierung,
    Entfernung von Satzzeichen und Standardisierung häufiger
    Geschäftsbegriffe für zuverlässige Ground-Truth-Vergleiche.
    
    Args:
        s (str | None): Zu normalisierender Text
        
    Returns:
        str: Kanonisierter Text in Kleinbuchstaben ohne Satzzeichen
        
    Note:
        - Normalisiert Unicode-Zeichen (z.B. Umlaute)
        - Standardisiert Konjunktionen (&, und → and)
        - Entfernt Rechtsformen (GmbH, AG, etc.)
        - Kollabiert Leerzeichen
    """
    if not s:
        return ""
    # Normalize unicode characters (e.g., umlauts) and case
    s = unicodedata.normalize("NFKD", str(s).lower())

    # Standardize conjunctions and common abbreviations
    s = re.sub(r'\s*&\s*|\s*und\s*', ' and ', s)
    s = re.sub(r'\bgmbh\.?|\bgesellschaft mit beschränkter haftung', '', s)
    s = re.sub(r'\bag\.?|\baktiengesellschaft', '', s)

    # Remove all non-alphanumeric characters except for spaces
    s = ''.join(c for c in s if c.isalnum() or c.isspace())

    # Collapse multiple spaces into one and remove leading/trailing whitespace
    return ' '.join(s.split()).strip()


def is_name_match(expected: str | None, predicted: str | None) -> bool:
    """
    Führt einen nachsichtigen Vergleich für Namen durch.
    
    Diese Funktion überprüft, ob die Kernwörter des vorhergesagten Namens
    im erwarteten Namen enthalten sind. Sie ist toleranter gegenüber
    Variationen in Firmennamen und berücksichtigt, dass OCR/LLM oft
    nur Teile von vollständigen Namen extrahiert. Dies wiederspiegelt der Prozess
    der Findung des Buchungskreis im nachgelagerten ERPSystem. 
    
    Args:
        expected (str | None): Erwarteter Name aus Ground Truth
        predicted (str | None): Vorhergesagter Name aus Pipeline
        
    Returns:
        bool: True wenn vorhergesagte Wörter Teilmenge der erwarteten sind
        
    """
    if not expected or not predicted:
        return not expected and not predicted

    canon_expected = set(canon_text(expected).split())
    canon_predicted = set(canon_text(predicted).split())

    # Check if the set of predicted words is a subset of the expected words
    return canon_predicted.issubset(canon_expected)

def _canon_id(s: str | None) -> str:
    """
    Kanonisiert einen ID-String für robusten Vergleich.
    
    Entfernt alle Nicht-Alphanumerischen Zeichen und konvertiert
    zu Großbuchstaben für einheitliche USt-ID und IBAN-Vergleiche.
    
    Args:
        s (str | None): Zu kanonisierender ID-String
        
    Returns:
        str: Bereinigte ID nur mit Großbuchstaben und Zahlen
    """
    if not s: return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def is_match(field: str, true_val, pred_val) -> bool:
    """
    Vergleicht wahre und vorhergesagte Werte basierend auf dem Feldtyp.
    
    Diese Funktion wählt die passende Vergleichsstrategie basierend
    auf dem Feldtyp: Namen (nachsichtig), Geldbeträge (numerisch),
    IDs (alphanumerisch) oder allgemeiner Text (kanonisch).
    
    Args:
        field (str): Name des zu vergleichenden Feldes
        true_val: Wahrer Wert aus Ground Truth
        pred_val: Vorhergesagter Wert aus Pipeline
        
    Returns:
        bool: True bei Übereinstimmung nach feldspezifischen Regeln
        
    Feldtypen:
        - NAME_KEYS: Nachsichtiger Namensvergleich
        - MONEY_KEYS: Numerischer Vergleich mit Rundung
        - ID_KEYS: Alphanumerischer Vergleich ohne Sonderzeichen
        - Andere: Kanonischer Textvergleich
    """
    ID_KEYS = {"ust-id", "iban"}
    MONEY_KEYS = {"total_amount", "tax_rate"}
    NAME_KEYS = {"vendor_name", "recipient_name"}

    if true_val in (None, "", "null"): return pred_val in (None, "", "null")

    # Use the more forgiving comparison for name fields
    if field in NAME_KEYS:
        return is_name_match(true_val, pred_val)

    if field in MONEY_KEYS: return canon_number(true_val) == canon_number(pred_val)
    if field in ID_KEYS: return _canon_id(true_val) == _canon_id(pred_val)
    return canon_text(true_val) == canon_text(pred_val)


def check_success(gt: Dict, pred: Dict) -> bool:
    """
    Überprüft ob extrahierte Daten Geschäftsregeln für Automatisierung erfüllen.
    
    Diese Funktion implementiert die Geschäftslogik zur Bestimmung, ob eine
    Rechnung automatisch verarbeitet werden kann. Sie folgt zwei Hauptregeln
    basierend auf dem Vorhandensein einer Bestellnummer.
    
    Args:
        gt (Dict): Ground Truth Daten
        pred (Dict): Vorhergesagte Daten aus Pipeline
        
    Returns:
        bool: True wenn alle erforderlichen Felder korrekt extrahiert wurden
        
    Geschäftsregeln:
        - Mit Bestellnummer: Empfänger + Rechnungsdatum erforderlich
        - Ohne Bestellnummer: Empfänger + Rechnungsnummer + Datum + Betrag + 
          Währung + (IBAN oder USt-ID) erforderlich
    """
    # Rule 1: With Purchase Order Number
    if is_match("purchase_order_number", gt.get("purchase_order_number"), pred.get("purchase_order_number")):
        return all([
            is_match("recipient_name", gt.get("recipient_name"), pred.get("recipient_name")),
            is_match("invoice_date", gt.get("invoice_date"), pred.get("invoice_date"))
        ])

    # Rule 2: Without Purchase Order Number
    else:
        # Check if at least one of iban or ust-id is a match
        id_match = any([
            is_match("iban", gt.get("iban"), pred.get("iban")),
            is_match("ust-id", gt.get("ust-id"), pred.get("ust-id"))
        ])

        return all([
            is_match("recipient_name", gt.get("recipient_name"), pred.get("recipient_name")),
            is_match("invoice_number", gt.get("invoice_number"), pred.get("invoice_number")),
            is_match("invoice_date", gt.get("invoice_date"), pred.get("invoice_date")),
            is_match("total_amount", gt.get("total_amount"), pred.get("total_amount")),
            is_match("currency", gt.get("currency"), pred.get("currency")),
            id_match
        ])