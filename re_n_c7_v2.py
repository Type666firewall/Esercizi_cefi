'''
Di seguito uno script che ti aiuterà a identificare e mettere in evidenza gli outlier nel dataset `diabetes.csv`.
Utilizzeremo il metodo dell'Interquartile Range (IQR) per identificare gli outlier e visualizzarli con un box plot. 
Questo script utilizza le librerie `pandas`, `numpy` e `matplotlib` per la visualizzazione.

Assicurati di avere installato le librerie necessarie nel tuo ambiente Python. Puoi installarle utilizzando pip, se necessario:

```bash
pip install pandas numpy matplotlib seaborn
```

### Script per identificare e visualizzare gli outlier
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
file_path = r'C:\Users\utente\Desktop\Antonio Giordano\python\Diabetes.csv'  # Cambia il percorso se necessario
data = pd.read_csv(file_path)

# Visualizza le prime righe del dataset
print(data.head())

# Seleziona la colonna da analizzare (ad esempio 'Glucose')
column_to_analyze = 'glucose'  # Sostituisci con la colonna che desideri analizzare

# Calcola Q1 e Q3
Q1 = data[column_to_analyze].quantile(0.25)
Q3 = data[column_to_analyze].quantile(0.75)
IQR = Q3 - Q1

# Definisci i limiti per gli outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifica gli outlier
outliers = data[(data[column_to_analyze] < lower_bound) | (data[column_to_analyze] > upper_bound)]

# Stampa il numero di outlier trovati
print(f"Numero di outlier trovati nella colonna {column_to_analyze}: {outliers.shape[0]}")

# Visualizza i dati con un boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=data[column_to_analyze], color='lightblue')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.title(f'Box Plot per {column_to_analyze} con Outlier evidenziati')
plt.legend()
plt.show()

# Visualizza gli outlier
print("Outlier trovati:")
print(outliers)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='glucose', y='BMI', hue='diabetes', style=None)
plt.title('Scatter Plot di glucose vs BMI con gli Outlier evidenziati')
plt.axhline(y=upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.axhline(y=lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.show()
print(f"Numero di outlier trovati nella colonna {'BMI'}: {outliers.shape[0]}")
plt.figure(figsize=(12, 6))
sns.scatterplot(data=outliers, x='glucose', y='BMI', color='black', label='Outliers')
sns.scatterplot(data=data, x='glucose', y='BMI', color='red', alpha=0.5, label='Altri Dati')
plt.title('Outlier nella Glicemia rispetto al BMI')
plt.xlabel('Glicemia')
plt.ylabel('BMI')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x=data['glucose'])
plt.title('Boxplot della Glicemia per Identificare gli Outlier')
plt.xlabel('Glicemia')
plt.show()
# --- Rimozione degli outlier dalla colonna 'glucose' ---
# (opzionale, da fare se si vuole allenare il modello su dati "puliti")
data_cleaned = data[(data[column_to_analyze] >= lower_bound) & (data[column_to_analyze] <= upper_bound)]

print(f"Dati originali: {data.shape[0]} righe")
print(f"Dati dopo rimozione outlier su '{column_to_analyze}': {data_cleaned.shape[0]} righe")

# --- Preprocessing ---
# Rinomina colonna 'diabetes' se serve (dipende dal dataset, spesso è 'Outcome')
target_column = 'diabetes' if 'diabetes' in data.columns else 'Outcome'

# Separazione in X e y
X = data_cleaned.drop(columns=[target_column])
y = data_cleaned[target_column]

# Suddivisione in training e test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento del modello
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predizione e valutazione
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("\n--- Valutazione del Modello ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Matrice di Confusione:")
print(confusion_matrix(y_test, y_pred))
print("\nReport di Classificazione:")
print(classification_report(y_test, y_pred))

# Nuova osservazione
x_prova = {
    'pregnancies': [5],
    'glucose': [112],
    'diastolic': [62],
    'triceps': [39],
    'insulin': [31],
    'BMI': [27.6],
    'dpf': [0.497],
    'age': [33]
}

# Converti in DataFrame
df_prova = pd.DataFrame(x_prova)

# Verifica colonne corrispondenti
print("Dati in input:")
print(df_prova)

# Predizione
y_pred = model.predict(df_prova)
print(f"\nPredizione (0 = non diabetico, 1 = diabetico): {y_pred[0]}")


### Spiegazione dello Script
'''
1. **Caricamento dei Dati**: Carica il dataset `diabetes.csv` e mostra le prime righe per un controllo visivo.

2. **Calcolo degli Outlier**: Utilizza il metodo dell'Interquartile Range (IQR) per calcolare i limiti inferiori e superiori.
Gli outlier sono definiti come valori al di sotto del limite inferiore o al di sopra del limite superiore.

3. **Visualizzazione**: Crea un box plot della colonna selezionata (in questo caso 'Glucose') e segna i limiti inferiori e 
superiori con linee tratteggiate.

4. **Output**: Stampa il numero di outlier trovati e visualizza i dati di questi outlier.

### Modifica della Colonna da Analizzare
Puoi cambiare la variabile `column_to_analyze` con qualsiasi colonna presente nel tuo dataset per identificare e mettere in
evidenza gli outlier in quella colonna specifica.

### Esecuzione
Salva lo script in un file Python e eseguilo. Assicurati che il file `diabetes.csv` si trovi nel percorso corretto specificato 
nel codice.
--------------------------------------------------
I risultati....
Lo script ha identificato gli outlier nella colonna **Glucose** del tuo dataset. Ora, approfondiamo un po’ di più su cosa significano 
questi outlier e come puoi analizzarli ulteriormente.

### Cosa Significano gli Outlier

Gli outlier, in questo caso, sono valori della glicemia (Glucose) che si discostano significativamente dalla maggior parte degli altri 
dati nel dataset. La loro presenza può suggerire vari scenari:

1. **Errori di Misurazione**: Potrebbero rappresentare misurazioni errate o malfunzionamenti nei dispositivi utilizzati per testare i 
  livelli di glucosio.

2. **Casi Clinici Speciali**: Potrebbero anche indicare pazienti con condizioni mediche particolari, come il diabete mellito o altre 
 patologie metaboliche, che presentano livelli di glucosio insolitamente elevati o bassi.

3. **Influenza sui Modelli di Predizione**: Gli outlier possono influenzare significativamente i risultati dei modelli di regressione e 
 di machine learning, quindi è importante valutarli con attenzione.

### Analisi Approfondita degli Outlier

Ecco alcuni suggerimenti su come procedere con l'analisi degli outlier identificati:

1. **Esamina gli Outlier**: Stampa o visualizza i dati degli outlier per vedere se ci sono schemi o motivi ricorrenti. 
Puoi usare il codice seguente per visualizzarli:

   ```python
   print("Outlier trovati:")
   print(outliers)
   ```

2. **Visualizzazione Aggiuntiva**: Puoi utilizzare un grafico a dispersione (scatter plot) per vedere come gli outlier si distribuiscono 
 rispetto ad altre variabili nel dataset, come l'**Outcome** (risultato), che indica la presenza di diabete.

   ```python
   plt.figure(figsize=(10, 6))
   sns.scatterplot(data=data, x='Glucose', y='BMI', hue='Outcome', style='Outcome')
   plt.title('Scatter Plot di Glucose vs BMI con gli Outlier evidenziati')
   plt.axhline(y=upper_bound, color='green', linestyle='--', label='Upper Bound')
   plt.axhline(y=lower_bound, color='red', linestyle='--', label='Lower Bound')
   plt.show()
   ```

3. **Considera di Non Eliminarli**: Poiché hai indicato che gli outlier sono importanti, potresti voler includerli nel tuo modello di regressione 
 o analisi predittiva. Tuttavia, prendi in considerazione che potrebbero influenzare le previsioni. Potresti provare a costruire il modello sia
  includendoli che escludendoli e confrontare i risultati.

4. **Analisi Statistica**: Calcola statistiche descrittive per gli outlier e confrontale con quelle dei dati normali per vedere se ci sono differenze
 significative.

5. **Classificazione o Raggruppamento**: Se gli outlier rappresentano un gruppo di pazienti con condizioni particolari, potresti voler considerarli in
un’analisi di clustering o in un modello di classificazione separato.

### Codice per Visualizzare gli Outlier

Ecco un esempio di codice che puoi utilizzare per visualizzare i dati degli outlier:

```python
# Visualizza i dati degli outlier
plt.figure(figsize=(12, 6))
sns.scatterplot(data=outliers, x='Glucose', y='BMI', color='red', label='Outliers')
sns.scatterplot(data=data, x='Glucose', y='BMI', color='blue', alpha=0.5, label='Altri Dati')
plt.title('Outlier nella Glicemia rispetto al BMI')
plt.xlabel('Glicemia')
plt.ylabel('BMI')
plt.legend()
plt.show()
```

Questo scatter plot ti mostrerà come gli outlier si comportano rispetto agli altri dati. Puoi sostituire **BMI** con qualsiasi altra 
colonna del tuo dataset per esplorare altre relazioni.
---------------------------------------------------------
Hai trovato un elenco di outlier nella colonna **Glucose** del tuo dataset. Analizziamo i dati identificati e cosa significano.

### Outlier Identificati

Ecco un riepilogo degli outlier che hai trovato:

| Index | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
|-------|-------------|---------|---------------|----------------|---------|------|-------------------------|-----|---------|
| 75    | 1           | 0       | 48            | 20             | 0       | 24.7 | 0.140                   | 22  | 0       |
| 182   | 1           | 0       | 74            | 20             | 23      | 27.7 | 0.299                   | 21  | 0       |
| 342   | 1           | 0       | 68            | 35             | 0       | 32.0 | 0.389                   | 22  | 0       |
| 349   | 5           | 0       | 80            | 32             | 0       | 41.0 | 0.346                   | 37  | 1       |
| 502   | 6           | 0       | 68            | 41             | 0       | 39.0 | 0.727                   | 41  | 1       |

### Interpretazione degli Outlier

1. **Glucose = 0**: 
   - I pazienti 75, 182, 342 e 502 hanno valori di **Glucose** pari a 0, il che è molto improbabile per una misurazione glicemica. 
     Potrebbe trattarsi di un errore di registrazione o di dati mancanti.
   - Questi casi meritano particolare attenzione. Potrebbero indicare che i pazienti non sono stati testati correttamente o che
     ci sono problemi nei dati.

2. **Blood Pressure e Skin Thickness**:
   - Alcuni di questi pazienti hanno valori elevati di **BloodPressure** e **SkinThickness**, il che potrebbe indicare condizioni di salute preoccupanti o,
     nuovamente, anomalie nei dati.
   - In particolare, il paziente 349 ha una **BloodPressure** di 80 e una **SkinThickness** di 32, che potrebbe richiedere un'analisi più approfondita.

3. **Insulin**: 
   - Solo il paziente 182 ha un valore di **Insulin** diverso da zero, il che potrebbe indicare un'analisi incompleta per gli altri pazienti, i quali 
     potrebbero non aver ricevuto la somministrazione di insulina.

4. **Outcome**: 
   - Gli outlier mostrano vari esiti (0 o 1) relativi alla presenza o assenza di diabete. È interessante notare che i pazienti con **Glucose** a zero 
    sono tutti etichettati come 0, il che potrebbe significare che non hanno mostrato segni di diabete, ma ciò deve essere interpretato con cautela data
     l'anomalia nei dati.

### Passi Successivi

1. **Esamina i Dati**: Controlla la fonte dei dati per capire se questi valori anomali sono stati registrati correttamente.
2. **Visualizza gli Outlier**: Usa grafici a dispersione per mostrare dove questi outlier si collocano rispetto agli altri dati.
3. **Considera di Codificare i Dati Mancanti**: Potresti considerare di trattare i valori zero come dati mancanti piuttosto che valori validi, 
   se ciò è appropriato per la tua analisi.
4. **Rivaluta il Modello**: Includere o escludere questi outlier dai tuoi modelli potrebbe cambiare significativamente i risultati delle tue 
 analisi predittive.

### Codice per Visualizzare gli Outlier

Ecco un esempio di come visualizzare gli outlier nel tuo dataset:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizza gli outlier
plt.figure(figsize=(12, 6))
sns.boxplot(x=data['glucose'])
plt.title('Boxplot della Glicemia per Identificare gli Outlier')
plt.xlabel('Glicemia')
plt.show()
```

Il boxplot ti aiuterà a identificare visivamente dove si trovano gli outlier e a comprendere meglio la distribuzione dei dati.


----------------------------------------------------------------
Il grafico....


Un boxplot (o diagramma a scatola) è uno strumento molto utile per visualizzare la distribuzione di un dataset e identificare gli outlier.
Ecco una spiegazione dettagliata di cosa rappresenta un boxplot e come interpretarlo:

### Componenti di un Boxplot

1. **Scatola (Box)**:
   - Rappresenta l'intervallo interquartile (IQR), che contiene il 50% centrale dei dati. 
   - Il bordo inferiore della scatola corrisponde al **primo quartile (Q1)**, che rappresenta il valore al di sotto del quale si trova il 25% dei dati.
   - Il bordo superiore della scatola corrisponde al **terzo quartile (Q3)**, che rappresenta il valore al di sotto del quale si trova il 75% dei dati.
   - La lunghezza della scatola è l'IQR: \( \text{IQR} = Q3 - Q1 \).

2. **Linea Mediana**:
   - All'interno della scatola, una linea orizzontale rappresenta la **mediana** (Q2), il valore centrale del dataset.

3. **"Whiskers" (Bristoli)**:
   - Le linee che si estendono dalla parte superiore e inferiore della scatola sono chiamate **whiskers**. Indicano l'estensione 
     dei dati all'interno di un certo intervallo.
   - Di solito, i whiskers si estendono fino a 1.5 volte l'IQR oltre i quartili (Q1 e Q3). 
   - Se ci sono dati oltre questo intervallo, vengono considerati outlier.

4. **Outlier**:
   - I punti che si trovano al di fuori dei whiskers sono considerati **outlier**. Questi punti sono rappresentati da punti o cerchi 
   separati nel grafico.

### Interpretazione del Boxplot

Quando guardi il boxplot per la colonna **Glucose**:

1. **Centralità**:
   - La posizione della mediana ti dà un'idea di dove si concentra la maggior parte dei valori della glicemia.
   
2. **Dispersione**:
   - La lunghezza della scatola (IQR) mostra la variabilità dei valori. Una scatola lunga indica una maggiore dispersione dei dati,
    mentre una scatola corta indica che i dati sono più concentrati attorno alla mediana.

3. **Outlier**:
   - I punti oltre i whiskers ti mostrano quali valori sono considerati outlier. Nel tuo caso, i valori di glucosio a zero appaiono 
    come outlier nel boxplot, il che significa che sono molto distanti dalla maggior parte dei valori di glicemia nel dataset.

### Esempio di Boxplot

Ecco un esempio di come apparirebbe un boxplot per la colonna **Glucose**:

```
   +---+----+-------+---+---------+---+---+
  0|    |  * |       |   |         |   |   Outlier
   |    |    |       |   |         |   |
   |    |    |       |   |         |   |
   |    |    |       |   |         |   |
   |    |    |       |   |         |   |
   |    |    |       |   |         |   |
   |    |    |       |   |         |   |
   |    |    |-------|---|---------|---| 
   |    |    |       |   |         |   |
   |    |    |       |   |         |   |
   |    |    |       |   |         |   |
   +---+----+-------+---+---------+---+---+
      Q1         Q2 (mediana)        Q3
```

### Vantaggi dell'uso del Boxplot

- **Identificazione degli outlier**: Ti aiuta a individuare rapidamente valori estremi o anomali.
- **Visualizzazione della distribuzione**: Fornisce una chiara rappresentazione della distribuzione dei dati, della mediana e della variabilità.
- **Confronto tra gruppi**: Può essere utilizzato per confrontare le distribuzioni di diverse categorie (ad esempio, glicemia per pazienti con diabete e senza).

### Conclusione

Il boxplot è uno strumento prezioso per l'analisi esplorativa dei dati, specialmente quando si cerca di comprendere la distribuzione e la
 presenza di outlier. In questo caso, il boxplot per la colonna **Glucose** ti permette di evidenziare i pazienti con valori anomali di glicemia,
 aiutandoti a prendere decisioni informate sui dati e sulle analisi successive.
  
  
  # salvare il modello previsionale e testarlo su dati fittizzi di un nuovo paziente 
'''