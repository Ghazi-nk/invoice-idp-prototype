#!/usr/bin/env python3
"""
Umfassende Analyse der Benchmark-Ergebnisse für die IDP-Pipeline.

Dieses Skript kombiniert alle wichtigen Analyseschritte zur Erstellung
der finalen CSV-Ergebnisse für die wissenschaftliche Auswertung der
Bachelor-Arbeit. Es führt Dokumentenklassifikation, Musteranalyse
und wissenschaftliche Evaluation durch.

Generierte CSV-Dateien:
- complete_invoice_analysis.csv: Vollständige Rechnungsanalyse
- field_performance_analysis.csv: Feldspezifische Performance-Analyse

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import pandas as pd
from pathlib import Path
import warnings
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.logging_config import analysis_logger

warnings.filterwarnings('ignore')

class ComprehensiveResultsAnalyzer:
    """Umfassender Analyzer für alle Benchmark-Ergebnisse."""
    
    def __init__(self, benchmark_dir: str = None):
        """
        Initialisiert den Analyzer.
        
        Args:
            benchmark_dir: Pfad zum Benchmark-Verzeichnis
        """
        # Bestimme automatisch das richtige Verzeichnis
        if benchmark_dir is None:
            # Wir sind in app/benchmark
            current_dir = Path(__file__).parent
            self.benchmark_dir = current_dir
        else:
            self.benchmark_dir = Path(benchmark_dir)
            
        self.results_dir = self.benchmark_dir# / "results" todo: uncomment this
        
        # Lade Benchmark-Daten
        self.load_benchmark_data()
    
    def load_benchmark_data(self):
        """Lädt alle Benchmark-Daten."""
        analysis_logger.info("Lade Benchmark-Daten...")
        
        try:
            self.summary_df = pd.read_csv(self.results_dir / "summary_llama3.1_8b.csv")
            self.details_df = pd.read_csv(self.results_dir / "details_llama3.1_8b.csv")
            self.results_df = pd.read_csv(self.results_dir / "results_llama3.1_8b.csv")
            analysis_logger.info(f"Daten geladen: {len(self.summary_df)} Summary, {len(self.details_df)} Details, {len(self.results_df)} Results")
        except FileNotFoundError as e:
            analysis_logger.error(f"Fehler beim Laden der Daten: {e}")
            analysis_logger.error("Stelle sicher, dass die Benchmark-Ergebnisse vorhanden sind.")
            raise
    

    
    def create_complete_invoice_analysis(self):
        """Erstellt complete_invoice_analysis.csv mit vollständiger Analyse."""
        analysis_logger.info("Erstelle vollständige Rechnungsanalyse...")
        
        # Statistiken pro Rechnung direkt aus summary_df
        invoice_stats = self.summary_df.groupby('invoice').agg({
            'success': ['sum', 'count'],
            'accuracy': 'mean',
            'f1': 'mean',
            'page_count': 'first',
            'searchable': 'first'
        })
        
        invoice_stats.columns = ['success_count', 'total_pipelines', 'avg_accuracy', 'avg_f1', 'page_count', 'searchable']
        invoice_stats['success_rate'] = invoice_stats['success_count'] / invoice_stats['total_pipelines']
        
        # Export
        output_file = self.results_dir / "complete_invoice_analysis.csv"
        invoice_stats.reset_index().to_csv(output_file, index=False)
        analysis_logger.info(f"Gespeichert: {output_file}")
        
        return invoice_stats
    
    def create_field_performance_analysis(self):
        """Erstellt field_performance_analysis.csv mit feldspezifischer Analyse."""
        analysis_logger.info("Erstelle Feld-Performance-Analyse...")
        
        # Gruppiere nach Pipeline und Feld (nur verfügbare Spalten verwenden)
        field_performance = self.details_df.groupby(['pipeline', 'field']).agg({
            'match': 'mean'
        }).round(3)
        
        # Pivot für bessere Darstellung
        field_pivot = field_performance.reset_index().pivot(index='field', columns='pipeline', values='match')
        
        # Berechne Durchschnittswerte pro Feld
        field_pivot['average'] = field_pivot.mean(axis=1)
        field_pivot['best_pipeline'] = field_pivot.drop('average', axis=1).idxmax(axis=1)
        field_pivot['best_score'] = field_pivot.drop(['average', 'best_pipeline'], axis=1).max(axis=1)
        
        # Export
        output_file = self.results_dir / "field_performance_analysis.csv"
        field_pivot.to_csv(output_file)
        analysis_logger.info(f"Gespeichert: {output_file}")
        
        return field_pivot
    

    
    def generate_summary_statistics(self):
        """Generiert zusammenfassende Statistiken."""
        analysis_logger.info("Generiere Zusammenfassungsstatistiken...")
        
        # Pipeline-Performance
        pipeline_stats = self.results_df.set_index('pipeline')
        
        analysis_logger.info("PIPELINE-PERFORMANCE:")
        for pipeline in pipeline_stats.index:
            row = pipeline_stats.loc[pipeline]
            analysis_logger.info(f"{pipeline:15s}: Acc={row['mean_accuracy']:.3f}, F1={row['mean_f1']:.3f}, Success={row['success_rate']:.3f}")
        
        # Feld-Performance
        field_stats = self.details_df.groupby('field').agg({
            'match': 'mean'
        }).round(3).sort_values('match', ascending=False)
        
        analysis_logger.info("FELD-PERFORMANCE (Top 5):")
        for field in field_stats.head().index:
            row = field_stats.loc[field]
            analysis_logger.info(f"{field:20s}: Match={row['match']:.3f}")
        

    
    def run_complete_analysis(self):
        """Führt die komplette Analyse durch und erstellt alle CSV-Dateien."""
        analysis_logger.info("STARTE UMFASSENDE BENCHMARK-ANALYSE")
        
        try:
            # 1. Vollständige Rechnungsanalyse
            complete_analysis = self.create_complete_invoice_analysis()
            
            # 2. Feld-Performance-Analyse
            field_performance = self.create_field_performance_analysis()
            
            # 3. Zusammenfassende Statistiken
            self.generate_summary_statistics()
            
            analysis_logger.info("ANALYSE ERFOLGREICH ABGESCHLOSSEN")
            analysis_logger.info("Generierte CSV-Dateien:")
            analysis_logger.info(f"   - {self.results_dir}/complete_invoice_analysis.csv")
            analysis_logger.info(f"   - {self.results_dir}/field_performance_analysis.csv")
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"FEHLER WÄHREND DER ANALYSE: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Hauptfunktion zur Ausführung der kompletten Analyse."""
    analysis_logger.info("COMPREHENSIVE RESULTS ANALYSIS")
    analysis_logger.info("Erstellt alle finalen CSV-Ergebnisse für die Bachelor-Arbeit")
    
    # Erstelle Analyzer und führe Analyse durch
    analyzer = ComprehensiveResultsAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        analysis_logger.info("Alle Analysen erfolgreich abgeschlossen.")
        analysis_logger.info("Die CSV-Dateien sind bereit für die wissenschaftliche Auswertung.")
    else:
        analysis_logger.error("Analyse mit Fehlern beendet. Bitte Logs prüfen.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())