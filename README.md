# Projekt-Dokumentation: Software & Datenmanagement

Dieses Dokument bietet eine umfassende Übersicht über das Projekt.

---

## Datenmanagementplan

### Administrative Informationen

**Projekttitel:**
Konzeption und prototypische Implementierung einer KI-basierten und intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess

**Author*in:**
Ghazi Nakkash

**Betreurer*in:**
- Erstgutachter: Prof. Dr.-Ing. Alexander Stanik
- Zweitgutachter: Prof. Dr. Arif Wider
- Praxispartner: PROMOS consult GmbH

**Institution:**
Hochschule für Technik und Wirtschaft Berlin
ROR: <https://ror.org/01xzwj424>

**Projektbeschreibung:**
Dieses Projekt entwickelt eine prototypische Intelligent Document Processing (IDP)-Pipeline zur automatisierten Extraktion von Informationen aus Eingangsrechnungen. Die Pipeline nutzt verschiedene OCR-Engines zur Texterkennung und Large Language Models (LLMs) zur semantischen Datenextraktion. Das Projekt evaluiert die Effektivität und Effizienz dieses Ansatzes.

### Methodik

**Datenerhebung:**
Es wurden zwei Arten von Daten verwendet:

1.  **Öffentliches Test-Set:** Eine Sammlung von 6 Beispielrechnungen ([5 einseitig](https://easyfirma.net/rechnungsvorlagen), [1 zweiseitig](https://rechnungsverwalter.de/pdfs/Beispielrechnung_mehrseitig.pdf) mit unterschiedlichen Layouts wurde für die Entwicklung und Demonstration der Pipeline verwendet.

2.  **Vertraulicher Benchmark-Datensatz:** Für die finale Evaluierung wurden x reale, nicht-anonymisierte Rechnungen vom Praxispartner PROMOS consult GmbH zur Verfügung gestellt. Diese wurden im Rahmen der Arbeit manuell gelabelt, um die korrekten Daten für das Benchmarking zu erstellen.

**Zu beachtende Richtlinien:**
Der öffentliche Datensatz enthält keine sensiblen Daten. Der vertrauliche Benchmark-Datensatz unterliegt strengen Datenschutz- und Geheimhaltungsvereinbarungen mit dem Praxispartner und wird nicht veröffentlicht.


### Zeitraum
Die Erhebung der Rechnungen und Erstellung des Evaluationsdatensatzes erfolgte im Zeitraum von 15.06.2025 bis 22.07.2025.
Die Entwicklung der Pipeline und die Durchführung der Benchmarks fanden zwischen 27.06.2025 und 02.08.2025.
Erste  Interview mit dem IT-consulting Team der PROMOS consult fand am 24.06.2025
Zweite Interview mit dem IT-consulting Team der PROMOS consult fand am 11.07.2025
Die Interview mit dem Cloud Development Team der PROMOS consult fand am 01.07.2025

### Dateneigenschaften

**Art und Umfang der Daten:**
- **Rohdaten:** Rechnungen im PDF-Format.
- **Referenzdaten (Ground Truth):** Manuell erstellte JSON-Dateien mit den korrekten Werten für jede Rechnung.
- **Ergebnisdaten:** CSV-Dateien mit detaillierten Benchmark-Ergebnissen unter [Results](app/benchmark/results) (z.B. Accuracy, Precision, Recall, F1-Score, Success Rate, Processing Duration).
- **Programmcode:** Python-Skripte für die Pipeline, OCR, LLM-Interaktion und das Benchmarking.
- **Daten-Schema:** Eine [`IncomingInvoiceSchema.json`](resources/IncomingInvoiceSchema.json)-Datei, die die Zieldatenstruktur definiert.

### Ablage und Speicherung

**Ablageort:**
Alle öffentlichen Daten (Code, öffentliches Test-Set, diese Dokumentation) werden in diesem Repository gespeichert. Der vertrauliche Benchmark-Datensatz wird nicht öffentlich abgelegt.
**Ablagestruktur:**
- **wo wird gespeichert:** Das Projekt ist logisch in Ordner wie `app/` (Anwendungscode), `resources/` (öffentliche Daten), `tests/` (Tests) und `notebooks/` (Experimente) strukturiert.
- **Dateibenennungskonvention:** Ergebnisdateien folgen der Konvention `[typ]_[modell].csv` (z.B. `summary_llama3.1_8b.csv`).
- **Versionierung:** Die Versionierung von Code und Dokumentation erfolgt über Git. Wichtige Meilensteine werden durch Git-Tags markiert.
- **Backup-Strategie:** Regelmäßige Pushes zum zentralen GitHub-Repository dienen als primäres Backup.
- **Langzeitarchivierung:** Das öffentliche GitHub-Repository dient als primärer und dauerhafter Speicherort für den Code und die öffentlichen Testdaten. Es sind keine weiteren Archivierungsmaßnahmen auf anderen Plattformen geplant. Für den vertraulichen Datensatz ist keine Langzeitarchivierung vorgesehen; er verbleibt beim Praxispartner.

---

## Teil 2: Software-Dokumentation (README)

### Analyse der Ausgangssituation und Anforderungen

Die Konzeption des Prototyps basiert auf der Analyse des IST-Zustands beim Praxispartner PROMOS consult GmbH, die in Kapitel 3 der Bachelorarbeit dokumentiert ist. Die zentralen Erkenntnisse aus den Interviews mit den Stakeholdern (IT-Consulting-Team und Cloud-Development-Team) definieren die Leitplanken für dieses Projekt:

1.  **Qualität und Genauigkeit:** Die wichtigste Anforderung ist eine höchstmögliche Erkennungsgenauigkeit, um den manuellen Korrekturaufwand zu minimieren. Die schwankende und oft unzureichende Erkennungsqualität der bestehenden Lösung ist der zentrale Schmerzpunkt der Anwender.

2.  **Datensouveränität:** Die Verarbeitung sensibler Rechnungsdaten muss zwingend lokal auf den eigenen Servern erfolgen. Eine Weitergabe an externe Cloud-Dienste ist aufgrund von Datenschutzanforderungen der Kunden ausgeschlossen.

3.  **Nutzung vorhandener Systeme:** Als technische Vorgabe wurde die Einbindung des bestehenden, internen Ollama-Servers für alle LLM-basierten Aufgaben festgelegt. Dies dient der Standardisierung der Infrastruktur und der Nutzung bereits getätigter Investitionen.

Diese drei Punkte bilden das Fundament für die Architektur und die technologischen Entscheidungen, die im Rahmen dieses Prototyps getroffen wurden.

### Technologie-Stack & Werkzeuge

- **Programmiersprache:** Python (Version 3.12)
- **API-Framework:** FastAPI
- **LLM-Interaktion:** Ollama
- **Verwendetes LLM für Benchmarks:** Llama 3.1 8B
- **OCR-Ansätze:** Tesseract, PaddleOCR, EasyOCR, DocTR, LayoutLMv3

Die genauen Versionen aller Bibliotheken sind in der Datei [`requirements.txt`](requirements.txt) spezifiziert.

### Installation und Anwendung

#### Installation

1.  **Klone das Repository:**
    ```sh
    git clone [https://github.com/DEIN_BENUTZERNAME/DEIN_REPOSITORY.git](https://github.com/DEIN_BENUTZERNAME/DEIN_REPOSITORY.git)
    cd DEIN_REPOSITORY
    ```

2.  **Erstelle eine virtuelle Umgebung:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    ```

3.  **Installiere die Abhängigkeiten:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Konfiguriere die Umgebungsvariablen:**
    Erstelle eine `.env`-Datei und füge die notwendigen Konfigurationen hinzu.

#### Anwendung

**Starte den API-Server:**
```sh
    uvicorn app.api_server:app --reload
```



### Qualitätssicherung

1.  **Benchmarking gegen Ground Truth:** Die extrahierten Daten werden mit manuell verifizierten "Ground Truth"-JSONs verglichen.
2.  **Metriken:** Die Genauigkeit wird anhand von Accuracy, Precision, Recall und F1-Score bewertet (`app/benchmark/evaluation_utils.py`).

### Benchmark-Ergebnisse

Die Leistung der Pipeline wurde mit dem **vertraulichen Benchmark-Datensatz** evaluiert. Die folgende Tabelle zeigt eine Zusammenfassung der Ergebnisse bei Verwendung des `llama3.1_8b`-Modells.

| Pipeline | Accuracy | Precision | Recall | F1-Score | Ø Gesamtdauer (s) |
|:---------| :------- | :-------- | :----- | :------- | :---------------- |
| **x**    |
{Platzhalter für results}

**Hinweis:** Diese Ergebnisse sind aufgrund der Vertraulichkeit des Datensatzes nicht direkt reproduzierbar.

### Lizenzen und Datenschutz

- **Code-Lizenz:** Der Quellcode ist unter der **MIT-Lizenz** veröffentlicht.
- **Daten-Lizenz:** Das öffentliche Test-Set (`/resources/`) kann für Forschungs- und Testzwecke frei verwendet werden.
- **Datenschutz:** Der für die Benchmarks verwendete Datensatz ist streng vertraulich und wird nicht veröffentlicht.

### Veröffentlichung und Zitierung

Das Projekt ist öffentlich auf GitHub zugänglich. {füge zenodo doi ein für die gespeicherte repo version) 
