import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
import os

# Impostare i semi per la ripetibilità
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Forza operazioni deterministiche
tf.config.experimental.enable_op_determinism()

# Impostare l'uso di un solo thread (per evitare variabilità)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Se vuoi eseguire solo su CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Usa solo la CPU
# 2. Crea un dataset di esempio
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, n_clusters_per_class=1,
                           random_state=42)

# 3. Dividi in train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Definizione del modello
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 6. Compilazione del modello
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 7. Addestramento del modello
history = model.fit(X_train_scaled, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)

# 8. Valutazione del modello
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nPerdita sul set di test: {loss:.4f}")
print(f"Accuratezza sul set di test: {accuracy:.4f}")

# 9. Curva di addestramento
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Accuratezza (Training)')
plt.plot(history.history['val_accuracy'], label='Accuratezza (Validazione)')
plt.xlabel('Epoche')
plt.ylabel('Accuratezza')
plt.title('Curva di Addestramento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Previsioni sul test set
predictions = model.predict(X_test_scaled)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

print("\nPrime 10 Previsioni (Probabilità):")
print(predictions[:10].flatten())
print("\nPrime 10 Previsioni (Classi Binarie):")
print(binary_predictions[:10])
print("\nPrime 10 Etichette Reali:")
print(y_test[:10])

# 11. Valutazione più dettagliata
print("\nMatrice di Confusione:")
print(confusion_matrix(y_test, binary_predictions))

print("\nReport di Classificazione:")
print(classification_report(y_test, binary_predictions))
