import pandas as pd
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

# Функция для создания спектрограммы из аудиофайла
def create_spectrogram(file_name):
    try:
        y, sr = librosa.load(file_name, sr=None)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        ps_db = librosa.power_to_db(ps, ref=np.max)
        # Обеспечение единого размера спектрограммы для всех аудио
        if ps_db.shape[1] < 173:
            ps_db = np.pad(ps_db, ((0, 0), (0, 173 - ps_db.shape[1])), "constant")
        return ps_db[:, :173]  # Обрезаем или дополняем до размера 128x173
    except Exception as e:
        print(f"Ошибка при обработке файла {file_name}: {e}")
        return None

# Подготовка данных
data_df = pd.read_csv(r"C:\Users\DarkStell\Downloads\archive\UrbanSound8K.csv")
audio_dir = r'C:\Users\DarkStell\Downloads\archive'
X, y = [], []

for _, row in data_df.iterrows():
    file_path = os.path.join(audio_dir, f'fold{row["fold"]}', row['slice_file_name'])
    spectrogram = create_spectrogram(file_path)
    if spectrogram is not None:
        X.append(spectrogram)
        y.append(row['classID'])

X = np.array(X)[:, :, :, np.newaxis]  # Преобразование списка в 4D-массив
y = np.array(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X)
print(y)
# hub_layer = hub.KerasLayer("https://tfhub.dev/google/vggish/1", output_shape=[128], input_shape=[None, 128, 173, 1], dtype=tf.float32, trainable=False)
# hub_layer.save('saved_hub_layer', save_format='tf')
# Создание модели
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 173, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Предполагая, что у вас 10 классов
])


# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
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
