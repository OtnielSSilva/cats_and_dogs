import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Caminho para o dataset descompactado
PATH = 'cats_and_dogs'

# Exibindo a quantidade de imagens em cada categoria (treino e validação)
print(f"Total de gatos de treino: {len(os.listdir(os.path.join(PATH, 'train', 'cats')))}")
print(f"Total de cachorros de treino: {len(os.listdir(os.path.join(PATH, 'train', 'dogs')))}")
print(f"Total de gatos de validação: {len(os.listdir(os.path.join(PATH, 'validation', 'cats')))}")
print(f"Total de cachorros de validação: {len(os.listdir(os.path.join(PATH, 'validation', 'dogs')))}")


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Congelando as camadas do modelo base para não serem atualizadas no treinamento
base_model.trainable = False

# Criando o modelo final
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),       # Calcula a média global dos mapas de ativação
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Classificação binária: gato (0) ou cachorro (1)
])

model.summary()

# Compilando o modelo com um learning rate menor (para não "destruir" os pesos pré-treinados)
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)


# Data Augmentation: técnica para aumentar a quantidade de dados de treino
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),    # Compatível com MobileNetV2
    batch_size=32,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),    # Compatível com MobileNetV2
    batch_size=32,
    class_mode="binary"
)

# Cálculo correto para usar TODAS as imagens em cada época
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Treinamento com Early Stopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,             
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop]
)

# Plotando gráficos de Acurácia e Perda
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(acc))

plt.figure()
plt.plot(epochs_range, acc, "r", label="Acurácia de Treino")
plt.plot(epochs_range, val_acc, "b", label="Acurácia de Validação")
plt.title("Acurácia de Treino e Validação")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs_range, loss, "r", label="Perda de Treino")
plt.plot(epochs_range, val_loss, "b", label="Perda de Validação")
plt.title("Perda de Treino e Validação")
plt.legend()
plt.show()

# Teste do modelo
num_test = 30

for i in range(1, num_test + 1):
    caminho = os.path.join("cats_and_dogs", "test", f"{i}.jpg")
    img = image.load_img(caminho, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Ajuste manual, pois o modelo espera imagem normalizada

    classes = model.predict(x, batch_size=10)
    prob = classes[0][0]
    
    print(f"\nImagem {i}.jpg - Probabilidade: {prob:.4f}")
    if prob > 0.5:
        print(f"{i}.jpg é um Cachorro")
    else:
        print(f"{i}.jpg é um Gato")
