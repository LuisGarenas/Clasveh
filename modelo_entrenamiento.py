import tensorflow as tf
import keras_cv
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import kagglehub
import os
import shutil
import matplotlib.pyplot as plt
import math

# ==========================================================================
# 1. CONFIGURACIÓN FINAL CON CLASES SIMPLIFICADAS
# ==========================================================================
class Config:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    
    CLASSES_TO_MERGE_GRANDE = ['E_pesado', 'bus']
    NEW_CLASS_GRANDE = 'Grande'
    
    CLASSES_TO_MERGE_CAMIONETA = ['jeep', 'SUV']
    NEW_CLASS_CAMIONETA = 'Camioneta'

    FINAL_CLASSES = ['Grande', 'Camioneta', 'family sedan']
    NUM_CLASSES = len(FINAL_CLASSES)
    
    LR_START = 1e-4
    LR_FINETUNE = 5e-6
    
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 5

CONFIG = Config()

# ==========================================================================
# 2. PREPARACIÓN DE DATOS
# ==========================================================================
def reorganize_data(base_dir, classes_to_merge, new_class_name):
    new_class_path = os.path.join(base_dir, new_class_name)
    os.makedirs(new_class_path, exist_ok=True)
    print(f"Creando la clase '{new_class_name}' y moviendo archivos...")
    for class_name in classes_to_merge:
        original_class_path = os.path.join(base_dir, class_name)
        if not os.path.exists(original_class_path):
            print(f"Advertencia: La carpeta '{class_name}' no existe y será omitida.")
            continue
        for filename in os.listdir(original_class_path):
            shutil.move(os.path.join(original_class_path, filename), new_class_path)
        try: os.rmdir(original_class_path)
        except OSError: pass
    print(f"Fusión para '{new_class_name}' completada.")

try:
    path = kagglehub.dataset_download("marquis03/vehicle-classification")
    train_dir = os.path.join(path, 'train')
    reorganize_data(train_dir, CONFIG.CLASSES_TO_MERGE_GRANDE, CONFIG.NEW_CLASS_GRANDE)
    reorganize_data(train_dir, CONFIG.CLASSES_TO_MERGE_CAMIONETA, CONFIG.NEW_CLASS_CAMIONETA)
except Exception as e:
    print(f"Error: {e}"); exit()

# ==========================================================================
# 3. GENERADORES DE DATOS
# ==========================================================================
datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2, rotation_range=25, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.25,
    horizontal_flip=True, fill_mode='nearest'
)
train_generator = datagen.flow_from_directory(
    train_dir, target_size=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), batch_size=CONFIG.BATCH_SIZE,
    class_mode='categorical', classes=CONFIG.FINAL_CLASSES, subset='training', shuffle=True
)
validation_generator = datagen.flow_from_directory(
    train_dir, target_size=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), batch_size=CONFIG.BATCH_SIZE,
    class_mode='categorical', classes=CONFIG.FINAL_CLASSES, subset='validation', shuffle=False
)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))
print("\nPesos de clase para el nuevo problema:", class_weight_dict)

# ==========================================================================
# 4. CONSTRUCCIÓN DEL MODELO
# ==========================================================================
base_model = MobileNetV2(
    input_shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False
inputs = Input(shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.6)(x)
predictions = Dense(CONFIG.NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, predictions)

loss_function = keras_cv.losses.FocalLoss(from_logits=False, gamma=3.0)

# ==========================================================================
# 5. ENTRENAMIENTO Y AJUSTE FINO
# ==========================================================================
print("\n--- Fase 1: Entrenamiento de la cabeza ---")
optimizer_head = keras.optimizers.AdamW(learning_rate=CONFIG.LR_START, weight_decay=1e-5)
model.compile(optimizer=optimizer_head, loss=loss_function, metrics=['accuracy'])
callbacks_list = [
    EarlyStopping(monitor='val_loss', mode='min', patience=CONFIG.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=CONFIG.REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1)
]
history = model.fit(
    train_generator, epochs=150, validation_data=validation_generator,
    callbacks=callbacks_list, class_weight=class_weight_dict
)

print("\n--- Fase 2: Ajuste Fino ---")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
optimizer_finetune = keras.optimizers.AdamW(learning_rate=CONFIG.LR_FINETUNE, weight_decay=1e-6)
model.compile(optimizer=optimizer_finetune, loss=loss_function, metrics=['accuracy'])
if len(history.epoch) < 150:
    history_fine = model.fit(
        train_generator, epochs=len(history.epoch) + 150, initial_epoch=len(history.epoch),
        validation_data=validation_generator, callbacks=callbacks_list, class_weight=class_weight_dict
    )
else:
    history_fine = None

# ==========================================================================
# 6. VISUALIZACIÓN Y DIAGNÓSTICO
# ==========================================================================
print("\nGenerando gráficas del historial de entrenamiento...")
acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if history_fine:
    acc += history_fine.history.get('accuracy', [])
    val_acc += history_fine.history.get('val_accuracy', [])
    loss += history_fine.history.get('loss', [])
    val_loss += history_fine.history.get('val_loss', [])
epochs_range = range(len(acc))

if epochs_range:
    plt.figure(figsize=(14, 7)); plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precisión Entrenamiento'); plt.plot(epochs_range, val_acc, label='Precisión Validación')
    if history_fine: plt.axvline(x=len(history.epoch)-1, color='grey', linestyle='--', label='Inicio Ajuste Fino')
    plt.legend(loc='lower right'); plt.title('Precisión'); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Pérdida Entrenamiento'); plt.plot(epochs_range, val_loss, label='Pérdida Validación')
    if history_fine: plt.axvline(x=len(history.epoch)-1, color='grey', linestyle='--', label='Inicio Ajuste Fino')
    plt.legend(loc='upper right'); plt.title('Pérdida'); plt.grid(True)
    plt.savefig('training_history_simplified.png'); plt.show()

# ==========================================================================
# 7. GUARDADO FINAL DEL MODELO ORIGINAL
# ==========================================================================
model.save('modelo_simplificado_final.h5')
print('\nModelo original guardado como modelo_simplificado_final.h5')


# ==========================================================================
# 8. (NUEVO) CONVERSIÓN A TENSORFLOW LITE PARA DEPLOYMENT
# ==========================================================================
print("\n--- Convirtiendo el modelo a TensorFlow Lite con cuantización ---")

# Cargar el modelo recién guardado
saved_model = tf.keras.models.load_model('modelo_simplificado_final.h5', compile=False)

# Crear el convertidor desde el modelo Keras
converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)

# Activar la optimización por defecto (incluye cuantización)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Realizar la conversión
tflite_quant_model = converter.convert()

# Guardar el nuevo modelo optimizado
with open('modelo_optimizado.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print('\n✅ Modelo optimizado guardado como modelo_optimizado.tflite')
print(f"Tamaño original (.h5): {os.path.getsize('modelo_simplificado_final.h5') / 1e6:.2f} MB")
print(f"Tamaño optimizado (.tflite): {len(tflite_quant_model) / 1e6:.2f} MB")
