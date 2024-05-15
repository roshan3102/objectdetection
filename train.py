import os
import numpy as np
import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import categorical_crossentropy
#from keras.metric import categorical_accuracy

def load_data(image_folder, label_folder):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        # Preprocess the image (e.g., resize, normalize)
        # image = preprocess_image(image)
        images.append(image)
        
        label_path = os.path.join(label_folder, filename[:-4] + ".txt")
        try:
            with open(label_path, 'r') as file:
                label = file.readline().strip().split()
                # Convert label to floats
                label = [float(value) for value in label]
                print('Label',label)
                labels.append(label)
                print('Labels',labels)
        except FileNotFoundError:
            print(f"Label file '{label_path}' not found.")
            # Handle missing label file
            
    return np.array(images), np.array(labels)


def create_model(num_classes):
    model = Sequential()
    print("Sequential model created")
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1024, 1024, 3)))
    print('Fisrt Conv2D layer added')
    model.add(MaxPooling2D((2, 2)))
    print('First MaxPooling2D layer added')
    model.add(Conv2D(64, (3, 3), activation='relu'))
    print('Second Conv2D layer added')
    model.add(MaxPooling2D((2, 2)))
    print('Second MaxPooling2D layer added')
    model.add(Conv2D(64, (3, 3), activation='relu'))
    print('Third Conv2D layer added')
    model.add(Flatten())
    print('Flatten layer added')
    model.add(Dense(64, activation='relu'))
    print('Dense layer added')
    model.add(Dense(5, activation=None))
    print('Output layer added')
    return model

def custom_loss(y_true, y_pred):
    # Split the predictions into class probabilities and bounding box coordinates
    y_true_class = y_true[:, :5]
    y_true_bbox = y_true[:, 5:]

    y_pred_class = y_pred[:, :5]
    y_pred_bbox = y_pred[:, 5:]

    # Classification loss
    class_loss = categorical_crossentropy(y_true_class, y_pred_class)

    # Localization loss (smooth L1 loss)
    diff = tf.abs(y_true_bbox - y_pred_bbox)
    loc_loss = tf.where(diff < 1, 0.5 * diff ** 2, diff - 0.5)
    loc_loss = tf.reduce_sum(loc_loss, axis=1)
    # Weighted sum of classification and localization losses
    total_loss = class_loss + loc_loss

    return total_loss

def train_model(dataset_folder, label_folder, num_classes, types_of):
    print(types_of)
    images, labels = load_data(dataset_folder, label_folder)
    print('Labels',labels)
    print('Labels shape',labels.shape)
    model = create_model(num_classes)
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    model.fit(images, labels, epochs=10, batch_size=32)
    model.save('trained_model.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with specified dataset and labels.')
    parser.add_argument('--dataset', type=str, help='Path to dataset folder', required=True)
    parser.add_argument('--annot', type=str, help='Path to label folder', required=True)
    parser.add_argument('--classes', type=int, help='Number of classes', required=True)
    parser.add_argument('--labels', type=str, help='Types of classes', required=True)
    args = parser.parse_args()

    train_model(args.dataset, args.annot, args.classes, args.labels)
