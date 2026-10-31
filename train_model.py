import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Settings
img_size = 64
batch_size = 8

# ✅ Data Augmentation (IMPORTANT)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(img_size, img_size),
    color_mode='grayscale',   # ✅ important
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

print("Class indices:", train_data.class_indices)

val_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ✅ Better Model
model = models.Sequential([
    tf.keras.Input(shape=(64,64,1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Train longer
model.fit(train_data, validation_data=val_data, epochs=20)

# predict.py loads the HDF5 file. Keras 3 also keeps a .keras copy.
model.save("model.h5")
model.save("model.keras")

print("Model trained and saved!")