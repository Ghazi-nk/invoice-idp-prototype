"""
Intelligent Document Processing (IDP) Pipeline Package.

Dieses Paket implementiert eine vollständige IDP-Pipeline für die automatische
Verarbeitung und Extraktion von Rechnungsdaten aus PDF-Dokumenten.

Hauptkomponenten:
- OCR-Engines (Tesseract, PaddleOCR, EasyOCR, DocTR, LayoutLMv3)
- LLM-basierte semantische Extraktion (Ollama/Llama)
- Rule-based Post-Processing
- FastAPI REST-Interface
- Comprehensive Benchmarking
- Centralized Logging System

Autor: Bachelor-Arbeit IDP Prototype
"""

__version__ = "1.0.0"
__author__ = "Bachelor Thesis - IDP Prototype"

# Import der wichtigsten Komponenten für einfachen Zugriff
from .logging_config import (
    pipeline_logger, api_logger, ocr_logger, pdf_logger,
    semantic_logger, postprocessing_logger, benchmark_logger,
    analysis_logger, config_logger
)

__all__ = [
    'pipeline_logger', 'api_logger', 'ocr_logger', 'pdf_logger',
    'semantic_logger', 'postprocessing_logger', 'benchmark_logger', 
    'analysis_logger', 'config_logger'
]