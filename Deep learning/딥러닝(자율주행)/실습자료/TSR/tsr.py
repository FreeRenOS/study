import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
n_train = len(X_train)
n_validation = len(X_valid)

n_test = len(X_test)
image_shape = X_train[0].shape

n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import cv2
# Visualizations will be shown in the notebook.
#%matplotlib inline

readfile = pd.read_csv('signnames.csv')
sign_name = readfile['SignName'].values

train_classes, train_class_cnt = np.unique(y_train, return_counts = True)
test_classes, test_class_cnt = np.unique(y_test, return_counts = True)

fig, axis = plt.subplots(2,4, figsize=(15,6))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
axis = axis.ravel()
for i in range(8):
    idx = rnd.randint(0, n_train)
    img = X_train[idx]
    axis[i].axis('off')
    axis[i].set_title(sign_name[y_train[idx]])
    axis[i].imshow(img)


fig0 = plt.figure(figsize=(13,10))
plt.bar(np.arange(n_classes), train_class_cnt, align='center', label='train')
plt.bar(np.arange(n_classes), test_class_cnt, align='center', label='test')
plt.xlabel('Class: Name of Traffic sign')
plt.ylabel('Number of each class')
plt.xlim([-1, n_classes])
plt.xticks(np.arange(n_classes), sign_name, rotation=270)
plt.legend()
plt.tight_layout()

plt.show()

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def norm (img_data):
#     return (img_data - 128)/ 128
#     return img_data / np.max(img_data)
    return img_data / 255

def gray_scale(X):
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    X = X.reshape(X.shape + (1,))
    return X

X_train_gray = []
X_train_CLAHE = []
X_valid_gray = []
X_valid_CLAHE = []
X_test_gray = []
X_test_CLAHE = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
for i in range(n_train):
    X_train_gray.append(cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY))
    X_train_CLAHE.append(clahe.apply(X_train_gray[i]))
for i in range(n_validation):
    X_valid_gray.append(cv2.cvtColor(X_valid[i], cv2.COLOR_RGB2GRAY))
    X_valid_CLAHE.append(clahe.apply(X_valid_gray[i]))
for i in range(n_test):
    X_test_gray.append(cv2.cvtColor(X_test[i], cv2.COLOR_RGB2GRAY))
    X_test_CLAHE.append(clahe.apply(X_test_gray[i]))
    
fig, axis = plt.subplots(2,4, figsize=(15,6))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
axis = axis.ravel()

for i in range(8):
    idx = rnd.randint(0, n_train)
    img = X_train_CLAHE[idx]
    axis[i].axis('off')
    axis[i].set_title(sign_name[y_train[idx]])
    axis[i].imshow(img, 'gray')

plt.show()

X_train_arr = np.array(X_train_CLAHE)
X_valid_arr = np.array(X_valid_CLAHE)
X_test_arr = np.array(X_test_CLAHE)
X_train_arr = X_train_arr.reshape(X_train_arr.shape + (1,))
X_valid_arr = X_valid_arr.reshape(X_valid_arr.shape + (1,))
X_test_arr = X_test_arr.reshape(X_test_arr.shape + (1,))
X_train = norm(X_train_arr)
X_valid = norm(X_valid_arr)
X_test = norm(X_test_arr)

print(X_train.shape, y_train.shape)

import tensorflow as tf
print(tf.__version__)

y_train = tf.keras.utils.to_categorical(y_train, 43)
y_valid = tf.keras.utils.to_categorical(y_valid, 43)
y_test = tf.keras.utils.to_categorical(y_test, 43)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(96, (3,3), activation='relu', input_shape=(32, 32, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(43, activation='softmax')
])
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=1)
val_loss = model.evaluate(X_valid, y_valid)
test_loss = model.evaluate(X_test, y_test)

model.save("tsr_model.h5")