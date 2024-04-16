import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import os
from sklearn.model_selection import train_test_split

# Загрузка YAMNet из TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Функция для загрузки и преобразования аудиофайла
def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

# Подготовка данных
data_df = pd.read_csv(r"C:\Users\DarkStell\Downloads\archive\UrbanSound8K.csv")
audio_dir = r'C:\Users\DarkStell\Downloads\archive'
X, y = [], []

for _, row in data_df.iterrows():
    file_path = os.path.join(audio_dir, f'fold{row["fold"]}', row['slice_file_name'])
    audio = load_audio(file_path)
    scores, embeddings, spectrogram = yamnet_model(audio)
    embeddings = embeddings.numpy().mean(axis=0)  # Среднее значение по временной оси
    X.append(embeddings)
    y.append(row['classID'])

X = np.array(X)
y = np.array(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение простой модели на эмбеддингах, полученных от YAMNet
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test))
# Компиляция модели с новыми настройками
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Уменьшаем скорость обучения для fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Продолжаем обучение (fine-tuning)
history_fine = model.fit(X_train, y_train,
                         epochs=5,  # Дополнительное количество эпох
                         batch_size=32,
                         validation_data=(X_test, y_test))
# Оценка модели на тестовом наборе данных
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nТочность на тестовых данных: {test_accuracy*100:.2f}%")
# Сохранение модели
model.save('urban_sound_classification_model.h5')
print("Модель успешно сохранена.")
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Точность на обучении')
    plt.plot(epochs, val_acc, 'b', label='Точность на валидации')
    plt.title('Точность на обучении и валидации')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Потери на обучении')
    plt.plot(epochs, val_loss, 'b', label='Потери на валидации')
    plt.title('Потери на обучении и валидации')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Построение графиков
plot_training_history(history)
# Делаем предсказания на основе тестовых данных
predictions = model.predict(X_test)

# Выводим предсказанные классы (можно преобразовать в метки классов, если нужно)
predicted_classes = np.argmax(predictions, axis=1)
print("Примеры предсказанных классов:", predicted_classes[:10])

# Сравниваем с реальными классами
print("Реальные классы:", y_test[:10])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Предположим, что y_test содержит истинные метки классов, а predictions - предсказанные моделью метки классов
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Точность (Accuracy)
accuracy = accuracy_score(y_test, predicted_classes)
print(f"Точность: {accuracy}")

# Точность (Precision)
precision = precision_score(y_test, predicted_classes, average='weighted')  # Используйте 'macro' для невзвешенного среднего
print(f"Точность (Precision): {precision}")

# Полнота (Recall)
recall = recall_score(y_test, predicted_classes, average='weighted')  # Используйте 'macro' для невзвешенного среднего
print(f"Полнота (Recall): {recall}")

# F1-мера
f1 = f1_score(y_test, predicted_classes, average='weighted')
print(f"F1-мера: {f1}")

# Для более детального анализа можно вывести отчёт по классификации, который включает в себя точность, полноту и F1-меру для каждого класса
print(classification_report(y_test, predicted_classes))
