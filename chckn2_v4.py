import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFilter
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,TensorBoard
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
#import secrets



def visualize_samples(sample_paths, titles):
    plt.figure(figsize=(12, 6))
    for i in range(len(sample_paths)):
        plt.subplot(1, len(sample_paths), i+1)
        img = mpimg.imread(sample_paths[i])
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Directory paths
parent_path = r"C:\Users\pc\Desktop\mechatronics folder\projects\code\CNN_Model\dataFolder"
data_folders = [folder for folder in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, folder))]

# Visualize sample images from each class
sample_paths = []
titles = []

tensorboard_callback = TensorBoard(log_dir=r'C:\Users\pc\Desktop\mechatronics folder\projects\code\CNN_Model\dataFolder\logs', histogram_freq=1)

for folder in data_folders:
    folder_path = os.path.join(parent_path, folder)
    sample_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    sample_paths.append(sample_image_path)
    titles.append(folder)

visualize_samples(sample_paths, titles)

target_size=(128, 128)

def create_and_train_cnn(data_directory, target_size=(128, 128), epochs=1):
    train_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        validation_split=0.2
    )

    train_data = train_data_gen.flow_from_directory(
        data_directory,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',  
        subset="training"
    )

    validation_data = train_data_gen.flow_from_directory(
        data_directory,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',  
        subset="validation"
    )

    num_classes = len(train_data.class_indices)
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')  # Use num_classes instead of hardcoding 3
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        steps_per_epoch=len(train_data),
        validation_steps=len(validation_data),
        callbacks=[tensorboard_callback]

    )

    return model, train_data

model_path = os.path.join(r"C:\Users\pc\Desktop\mechatronics folder\projects\code\CNN_Model\chicken_model", 'chicken_predictor.tflite')

def train_model():
    if len(data_folders) == 3: 
        data_directory = parent_path
        trained_model,train_data = create_and_train_cnn(data_directory)
        trained_model.save(model_path)
        print("Model training completed and saved successfully.")
        return train_data
    else:
        print("Error: Expected three data folders, found", len(data_folders))
    

def get_train_data(data_directory=parent_path):
    train_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        validation_split=0.2
    )

    train_data = train_data_gen.flow_from_directory(
        data_directory,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',  
        subset="training"
    )
    return train_data


def predict(model_path, image_path):

    model = models.load_model(model_path)

    img = mpimg.imread(image_path) 
    img = Image.fromarray(img)  
    img = img.resize((target_size[0], target_size[1]))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0) 
    
    train_data = get_train_data()

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  
    predicted_proba = predictions[0][predicted_class]  

    class_names = list(train_data.class_indices.keys())
    predicted_class_name = class_names[predicted_class]


    return {
        "class": predicted_class_name,
        "probability": predicted_proba
    }


train_data = train_model()

test_data = r"C:\Users\pc\Desktop\mechatronics folder\projects\code\CNN_Model\test_data"
for img in os.listdir(test_data):
    feed = predict(model_path=model_path,image_path=os.path.join(test_data,img))
    clss,prob = feed["class"],feed["probability"]
    os.rename(os.path.join(test_data,img),os.path.join(test_data,f"{clss}-{prob}.jpg"))


