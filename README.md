# EDEAS Six Sigma AI Lab — Training Edition

Prototipo Streamlit funzionante per il Summary Report del dataset didattico Thickness.

## Avvio locale

1. Installa Python 3.11 o superiore.
2. Apri un terminale nella cartella del progetto.
3. Esegui `pip install -r requirements.txt`.
4. Esegui `streamlit run app.py`.
5. Apri l'indirizzo locale indicato dal terminale (di norma http://localhost:8501).

## Pubblicazione

Carica la cartella in un repository GitHub e pubblica tramite Streamlit Community Cloud selezionando `app.py`. Prima di distribuirla ai partecipanti, verifica privacy, accessi e termini del servizio di hosting.

## Note metodologiche

Il test Anderson-Darling e i relativi p-value sono calcolati da statsmodels. Gli intervalli di confidenza della deviazione standard presuppongono normalità. L'IC della mediana è non parametrico e conservativo. Gli algoritmi non sono garantiti identici a Minitab.

Questa prima versione non utilizza API di modelli AI: è il motore statistico deterministico su cui costruire successivamente l'agente.
