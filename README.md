# Analisi dei Crimini a Chicago (2001 - Presente)

Questo progetto utilizza **PySpark**, **Streamlit** e altre librerie per analizzare e visualizzare i dati sui crimini a Chicago dal 2001 ad oggi. Include funzionalità di esplorazione dei dati, analisi stagionali, modelli di machine learning e visualizzazioni interattive.

---

## Funzionalità Principali

### 1. **Esplorazione dei Dati**
- **Visualizzazione dei dati puliti**: Mostra il dataset dopo la rimozione dei valori nulli e il pre-processing.
- **Distribuzione dei crimini per tipo**: Grafici a barre che mostrano i crimini più comuni.
- **Distribuzione dei crimini per distretto**: Grafici orizzontali che mostrano il numero di crimini per distretto.

### 2. **Analisi Temporale**
- **Media mobile**: Mostra la media mobile dei crimini per mese.
- **Analisi stagionale**: Grafici che mostrano il numero di crimini per stagione (Inverno, Primavera, Estate, Autunno).
- **Trend giornaliero**: Analisi delle ore critiche in cui si verificano più crimini.

### 3. **Machine Learning**
- **Regressione Logistica**: Modello per prevedere gli arresti.
- **Random Forest**: Modello per classificare i crimini.
- **Gradient Boosting**: Modello per prevedere il numero di crimini.
- **Matrice di correlazione**: Visualizza le correlazioni tra le variabili.

### 4. **Visualizzazioni Geografiche**
- **Mappa interattiva**: Mostra i crimini per distretto con cluster e colori basati sulla gravità.
- **GeoJSON**: Conversione dei dati geografici in formato GeoJSON per l'uso con Folium.

---

