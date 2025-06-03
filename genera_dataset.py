#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per la generazione di un dataset fittizio che analizza la correlazione 
tra reddito e tasso di mortalità.

MIT License

Copyright (c) 2025 Antonio Giordano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Impostazione del seed per la riproducibilità
np.random.seed(42)

def genera_dataset(n_samples=1000, save_path='dataset_reddito_mortalita.csv'):
    """
    Genera un dataset fittizio che simula la relazione tra reddito e tasso di mortalità.
    
    Parametri:
    ----------
    n_samples : int, default=1000
        Numero di campioni da generare
    save_path : str, default='dataset_reddito_mortalita.csv'
        Percorso dove salvare il dataset generato
        
    Returns:
    --------
    pandas.DataFrame
        Il dataset generato
    """
    print(f"Generazione di un dataset con {n_samples} campioni...")
    
    # Generazione del reddito (in migliaia di euro)
    # Distribuzione log-normale per simulare la distribuzione del reddito nella popolazione
    reddito_medio = np.random.lognormal(mean=3.5, sigma=0.6, size=n_samples)

    # Aggiunta di variabili correlate al reddito e alla mortalità
    # Età: distribuzione normale tra 18 e 90 anni
    eta = np.random.normal(loc=45, scale=15, size=n_samples)
    eta = np.clip(eta, 18, 90)  # Limitiamo l'età tra 18 e 90 anni

    # Il livello di istruzione è correlato positivamente con il reddito
    # 0: Elementare, 1: Media, 2: Superiore, 3: Laurea, 4: Post-laurea
    istruzione_base = 0.5 + 0.5 * np.log1p(reddito_medio) / np.log1p(reddito_medio.max())
    istruzione = np.random.normal(loc=istruzione_base * 4, scale=0.8, size=n_samples)
    istruzione = np.clip(istruzione, 0, 4).astype(int)

    # Accesso alle cure mediche (0-10) correlato positivamente con il reddito
    accesso_cure_base = 3 + 7 * np.log1p(reddito_medio) / np.log1p(reddito_medio.max())
    accesso_cure = np.random.normal(loc=accesso_cure_base, scale=1.5, size=n_samples)
    accesso_cure = np.clip(accesso_cure, 0, 10)

    # Stile di vita salutare (0-10) correlato positivamente con reddito e istruzione
    stile_vita_base = 2 + 4 * np.log1p(reddito_medio) / np.log1p(reddito_medio.max()) + istruzione * 0.5
    stile_vita = np.random.normal(loc=stile_vita_base, scale=1.8, size=n_samples)
    stile_vita = np.clip(stile_vita, 0, 10)

    # Inquinamento ambientale (0-10) correlato negativamente con il reddito
    inquinamento_base = 8 - 5 * np.log1p(reddito_medio) / np.log1p(reddito_medio.max())
    inquinamento = np.random.normal(loc=inquinamento_base, scale=2.0, size=n_samples)
    inquinamento = np.clip(inquinamento, 0, 10)

    # Generazione del tasso di mortalità (per 1000 abitanti)
    # Fattori che aumentano il tasso di mortalità: età, inquinamento
    # Fattori che diminuiscono il tasso di mortalità: reddito, istruzione, accesso alle cure, stile di vita salutare

    # Componente base del tasso di mortalità
    mortalita_base = 2.0 + 0.15 * (eta - 18)  # Aumenta con l'età

    # Effetto del reddito (effetto protettivo)
    effetto_reddito = -3.0 * np.log1p(reddito_medio) / np.log1p(reddito_medio.max())

    # Effetto dell'istruzione (effetto protettivo)
    effetto_istruzione = -1.0 * istruzione / 4.0

    # Effetto dell'accesso alle cure (effetto protettivo)
    effetto_cure = -2.0 * accesso_cure / 10.0

    # Effetto dello stile di vita (effetto protettivo)
    effetto_stile_vita = -2.0 * stile_vita / 10.0

    # Effetto dell'inquinamento (effetto dannoso)
    effetto_inquinamento = 2.0 * inquinamento / 10.0

    # Calcolo del tasso di mortalità combinando tutti gli effetti
    tasso_mortalita = mortalita_base + effetto_reddito + effetto_istruzione + effetto_cure + effetto_stile_vita + effetto_inquinamento

    # Aggiunta di rumore casuale
    tasso_mortalita += np.random.normal(loc=0, scale=1.0, size=n_samples)

    # Assicuriamo che il tasso di mortalità sia positivo
    tasso_mortalita = np.clip(tasso_mortalita, 0.5, 30)

    # Creazione del DataFrame
    data = pd.DataFrame({
        'reddito': reddito_medio,
        'eta': eta,
        'istruzione': istruzione,
        'accesso_cure': accesso_cure,
        'stile_vita': stile_vita,
        'inquinamento': inquinamento,
        'tasso_mortalita': tasso_mortalita
    })

    # Aggiunta di una colonna categorica per il livello di istruzione
    livelli_istruzione = ['Elementare', 'Media', 'Superiore', 'Laurea', 'Post-laurea']
    data['livello_istruzione'] = data['istruzione'].map(lambda x: livelli_istruzione[x])

    # Aggiunta di una colonna per la fascia di reddito
    bins = [0, 20, 40, 60, 80, float('inf')]
    labels = ['Molto basso', 'Basso', 'Medio', 'Alto', 'Molto alto']
    data['fascia_reddito'] = pd.cut(data['reddito'], bins=bins, labels=labels)

    # Aggiunta di una colonna per la fascia d'età
    bins_eta = [18, 30, 45, 60, 75, 90]
    labels_eta = ['18-30', '31-45', '46-60', '61-75', '76-90']
    data['fascia_eta'] = pd.cut(data['eta'], bins=bins_eta, labels=labels_eta)

    # Salvataggio del dataset
    data.to_csv(save_path, index=False)
    print(f"Dataset generato e salvato con successo in: {save_path}")
    print(f"Dimensioni del dataset: {data.shape}")
    
    return data

def visualizza_correlazioni(data):
    """
    Visualizza le principali correlazioni nel dataset.
    
    Parametri:
    ----------
    data : pandas.DataFrame
        Il dataset da analizzare
    """
    # Creazione della directory per le immagini se non esiste
    if not os.path.exists('immagini'):
        os.makedirs('immagini')
    
    # Calcolo della matrice di correlazione
    corr_matrix = data[['reddito', 'eta', 'istruzione', 'accesso_cure', 
                        'stile_vita', 'inquinamento', 'tasso_mortalita']].corr()
    
    print("\nMatrice di correlazione:")
    print(corr_matrix)
    
    # Visualizzazione della matrice di correlazione
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matrice di Correlazione')
    plt.tight_layout()
    plt.savefig('immagini/matrice_correlazione.png')
    
    # Visualizzazione della relazione tra reddito e tasso di mortalità
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='reddito', y='tasso_mortalita', hue='fascia_eta', data=data)
    plt.title('Relazione tra Reddito e Tasso di Mortalità')
    plt.xlabel('Reddito (migliaia di €)')
    plt.ylabel('Tasso di Mortalità (per 1000 abitanti)')
    plt.savefig('immagini/relazione_reddito_mortalita.png')
    
    print("Visualizzazioni salvate nella directory 'immagini'")

if __name__ == "__main__":
    # Generazione del dataset
    data = genera_dataset()
    
    # Visualizzazione delle statistiche descrittive
    print("\nStatistiche descrittive:")
    print(data.describe())
    
    # Visualizzazione delle correlazioni
    visualizza_correlazioni(data)
