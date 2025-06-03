#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per l'implementazione di un modello Random Forest Regressor per analizzare
la correlazione tra reddito e tasso di mortalità.

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import joblib

def carica_dataset(file_path='dataset_reddito_mortalita.csv'):
    """
    Carica il dataset dal file CSV.
    
    Parametri:
    ----------
    file_path : str, default='dataset_reddito_mortalita.csv'
        Percorso del file CSV contenente il dataset
        
    Returns:
    --------
    pandas.DataFrame
        Il dataset caricato
    """
    print(f"Caricamento del dataset da: {file_path}")
    data = pd.read_csv(file_path)
    print(f"Dataset caricato con successo. Dimensioni: {data.shape}")
    return data

def prepara_dati(data):
    """
    Prepara i dati per il modello di machine learning.
    
    Parametri:
    ----------
    data : pandas.DataFrame
        Il dataset da preparare
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    print("Preparazione dei dati per il modello...")
    
    # Selezione delle features e del target
    X = data[['reddito', 'eta', 'istruzione', 'accesso_cure', 'stile_vita', 'inquinamento']]
    y = data['tasso_mortalita']
    
    # Divisione in set di training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dati divisi in: {X_train.shape[0]} campioni di training e {X_test.shape[0]} campioni di test")
    
    # Standardizzazione delle features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def addestra_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Addestra un modello Random Forest Regressor.
    
    Parametri:
    ----------
    X_train : array-like
        Features di training
    y_train : array-like
        Target di training
    n_estimators : int, default=100
        Numero di alberi nel forest
    random_state : int, default=42
        Seed per la riproducibilità
        
    Returns:
    --------
    RandomForestRegressor
        Il modello addestrato
    """
    print(f"Addestramento del modello Random Forest Regressor con {n_estimators} alberi...")
    
    # Creazione e addestramento del modello
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Validazione incrociata
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Punteggi di validazione incrociata (R²): {cv_scores}")
    print(f"Media dei punteggi di validazione incrociata (R²): {cv_scores.mean():.4f}")
    
    return model

def valuta_modello(model, X_test, y_test):
    """
    Valuta le performance del modello sul set di test.
    
    Parametri:
    ----------
    model : estimator
        Il modello da valutare
    X_test : array-like
        Features di test
    y_test : array-like
        Target di test
        
    Returns:
    --------
    dict
        Dizionario contenente le metriche di valutazione
    """
    print("Valutazione del modello sul set di test...")
    
    # Predizioni
    y_pred = model.predict(X_test)
    
    # Calcolo delle metriche
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Stampa delle metriche
    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R² (Coefficiente di determinazione): {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred
    }

def visualizza_risultati(model, X_test, y_test, y_pred, feature_names, scaler):
    """
    Visualizza i risultati del modello.
    
    Parametri:
    ----------
    model : estimator
        Il modello addestrato
    X_test : array-like
        Features di test
    y_test : array-like
        Target di test
    y_pred : array-like
        Predizioni del modello
    feature_names : array-like
        Nomi delle features
    scaler : StandardScaler
        Scaler utilizzato per standardizzare le features
    """
    # Creazione della directory per le immagini se non esiste
    if not os.path.exists('immagini'):
        os.makedirs('immagini')
    
    # Visualizzazione delle predizioni vs valori reali
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valori Reali')
    plt.ylabel('Predizioni')
    plt.title('Predizioni vs Valori Reali - Random Forest Regressor')
    plt.savefig('immagini/predizioni_random_forest.png')
    
    # Visualizzazione dei residui
    residui = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residui, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
    plt.xlabel('Predizioni')
    plt.ylabel('Residui')
    plt.title('Residui - Random Forest Regressor')
    plt.savefig('immagini/residui_random_forest.png')
    
    # Visualizzazione dell'importanza delle features
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances.sort_values().plot(kind='barh')
    plt.title('Importanza delle Features - Random Forest Regressor')
    plt.xlabel('Importanza')
    plt.tight_layout()
    plt.savefig('immagini/importanza_features.png')
    
    # Analisi dell'effetto del reddito sul tasso di mortalità
    print("\nAnalisi dell'effetto del reddito sul tasso di mortalità...")
    
    # Preparazione dei dati per l'analisi dell'effetto del reddito
    reddito_range = np.linspace(20, 100, 100)  # Range di reddito da analizzare
    
    # Valori medi delle altre features
    eta_media = 45
    istruzione_media = 2
    accesso_cure_medio = 6
    stile_vita_medio = 5
    inquinamento_medio = 5
    
    # Creazione di un dataset con reddito variabile e altre features fisse alla media
    X_reddito_var = np.column_stack([
        reddito_range,
        np.full_like(reddito_range, eta_media),
        np.full_like(reddito_range, istruzione_media),
        np.full_like(reddito_range, accesso_cure_medio),
        np.full_like(reddito_range, stile_vita_medio),
        np.full_like(reddito_range, inquinamento_medio)
    ])
    
    # Standardizzazione dei dati
    X_reddito_var_scaled = scaler.transform(X_reddito_var)
    
    # Predizione del tasso di mortalità
    y_pred_reddito = model.predict(X_reddito_var_scaled)
    
    # Visualizzazione dell'effetto del reddito sul tasso di mortalità
    plt.figure(figsize=(12, 8))
    plt.plot(reddito_range, y_pred_reddito, 'r-', linewidth=2)
    plt.xlabel('Reddito (migliaia di €)')
    plt.ylabel('Tasso di Mortalità Previsto (per 1000 abitanti)')
    plt.title('Effetto del Reddito sul Tasso di Mortalità\n(altre variabili fissate alla media)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('immagini/effetto_reddito_mortalita.png')
    
    print("Visualizzazioni salvate nella directory 'immagini'")

def salva_modello(model, scaler, file_path='modello_random_forest.pkl'):
    """
    Salva il modello addestrato e lo scaler in un file.
    
    Parametri:
    ----------
    model : estimator
        Il modello da salvare
    scaler : StandardScaler
        Lo scaler da salvare
    file_path : str, default='modello_random_forest.pkl'
        Percorso dove salvare il modello
    """
    # Creazione della directory per i modelli se non esiste
    if not os.path.exists('modelli'):
        os.makedirs('modelli')
    
    # Salvataggio del modello e dello scaler
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, os.path.join('modelli', file_path))
    print(f"Modello e scaler salvati in: modelli/{file_path}")

def main():
    """
    Funzione principale che esegue l'intero processo di analisi.
    """
    # Caricamento del dataset
    data = carica_dataset()
    
    # Preparazione dei dati
    X_train, X_test, y_train, y_test, scaler, feature_names = prepara_dati(data)
    
    # Addestramento del modello
    model = addestra_random_forest(X_train, y_train)
    
    # Valutazione del modello
    results = valuta_modello(model, X_test, y_test)
    
    # Visualizzazione dei risultati
    visualizza_risultati(model, X_test, y_test, results['y_pred'], feature_names, scaler)
    
    # Salvataggio del modello
    salva_modello(model, scaler)
    
    print("\nAnalisi completata con successo!")

if __name__ == "__main__":
    main()
