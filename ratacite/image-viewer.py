# Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
import os

# Load CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# save_dir = 'cifar10_images'
# os.makedirs(save_dir, exist_ok=True)

base_dir = "cifar10_resized"
IMG_SIZE = 224
os.makedirs(base_dir, exist_ok=True)

def save_resized_images(images, labels, subset):
    for i, (img, label) in enumerate(zip(images, labels)):
        class_name = class_names[label[0]]
        folder = os.path.join(base_dir, subset, class_name)
        os.makedirs(folder, exist_ok=True)
        img_resized = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]).numpy().astype('uint8')
        plt.imsave(os.path.join(folder, f"{i}.png"), img_resized)
    
save_resized_images(train_images, train_labels, 'train')
save_resized_images(test_images, test_labels, 'test')

# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary, interpolation='nearest')
#     plt.xlabel(class_names[train_labels[i][0]])
#     plt.imsave(f"{save_dir}/image_{i+1}_{class_names[train_labels[i][0]]}.png", train_images[i])
# plt.show()