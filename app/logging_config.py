"""
Zentrale Logging-Konfiguration für die IDP-Pipeline.

Dieses Modul stellt eine einheitliche Logging-Konfiguration für das gesamte
Intelligent Document Processing System bereit. Es definiert Logger-Instanzen
für verschiedene Komponenten und stellt sicher, dass alle Log-Nachrichten
konsistent formatiert und ausgegeben werden.

Logger-Hierarchie:
- idp: Haupt-Logger für Pipeline-Operationen
- idp.api: API-Server und Endpunkt-spezifische Logs
- idp.ocr: OCR-Engine-spezifische Logs
- idp.semantic: LLM und semantische Extraktion
- idp.postprocessing: Post-Processing und Verifikation
- idp.benchmark: Benchmark-System und Evaluation
- idp.analysis: Analyse-Skripte und Reporting

Log-Level:
- DEBUG: Detaillierte Entwicklungsinformationen
- INFO: Allgemeine Informationen über Programmablauf
- WARNING: Warnungen über potentielle Probleme
- ERROR: Fehler die das Programm beeinträchtigen
- CRITICAL: Schwerwiegende Fehler die zum Programmabbruch führen

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Logging-Konfiguration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> None:
    """
    Konfiguriert das zentrale Logging-System für die IDP-Pipeline.
    
    Args:
        level (int): Logging-Level (default: INFO)
        log_file (Optional[Path]): Pfad zur Log-Datei (optional)
        console_output (bool): Ob Logs auch auf der Konsole ausgegeben werden
        
    Note:
        Diese Funktion sollte einmal zu Beginn des Programms aufgerufen werden.
    """
    # Root-Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Bestehende Handler entfernen
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatter definieren
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # Console Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File Handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Erstellt oder gibt einen Logger mit dem spezifizierten Namen zurück.
    
    Args:
        name (str): Name des Loggers (z.B. 'idp.api', 'idp.ocr')
        
    Returns:
        logging.Logger: Konfigurierte Logger-Instanz
        
    Note:
        Logger folgen einer hierarchischen Namensstruktur mit 'idp' als Root.
    """
    return logging.getLogger(name)

# Vordefinierte Logger für verschiedene Komponenten
pipeline_logger = get_logger('idp.pipeline')
api_logger = get_logger('idp.api')
ocr_logger = get_logger('idp.ocr')
pdf_logger = get_logger('idp.pdf')
semantic_logger = get_logger('idp.semantic')
postprocessing_logger = get_logger('idp.postprocessing')
benchmark_logger = get_logger('idp.benchmark')
analysis_logger = get_logger('idp.analysis')
config_logger = get_logger('idp.config')

# Initialisiere Logging mit Standard-Konfiguration
setup_logging()