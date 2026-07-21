import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Path Lengkap ke Folder Training dan Validation
BASE_DIR = 'Final Dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'training')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'testing')

# 2. Data Augmentation untuk Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescale saja untuk Validation & Testing (Tanpa Augmentasi)
val_datagen = ImageDataGenerator(rescale=1./255)

print("--- Memuat Data Training ---")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("--- Memuat Data Validasi ---")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Catat urutan indeks kelas untuk Flask nanti
print("\nIndeks Kelas:", train_generator.class_indices)

# 3. Model Transfer Learning MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
num_classes = len(train_generator.class_indices)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# 4. Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Jalankan Training
EPOCHS = 10
print("\n=== Memulai Training Model ===")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 6. Evaluasi Opsional Menggunakan Folder Testing
if os.path.exists(TEST_DIR):
    print("\n--- Evaluasi Model dengan Data Testing ---")
    test_generator = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nAkurasi pada Data Testing: {test_acc * 100:.2f}%")

# 7. Simpan Model
MODEL_NAME = 'traffic_mobilenet.h5'
model.save(MODEL_NAME)
print(f"\n[SUKSES] Model berhasil disimpan sebagai '{MODEL_NAME}'!")