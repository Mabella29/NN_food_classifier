import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load data
train_dir = 'data/10_food_classes_10_percent/train'
test_dir = 'data/10_food_classes_10_percent/test'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Load VGG19 model
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

# Build model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=test_data,
    callbacks=[EarlyStopping(patience=5), ReduceLROnPlateau(factor=0.2, patience=3)]
)


model.save('models/vgg19_food_classifier.h5')