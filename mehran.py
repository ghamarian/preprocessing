from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from PIL import Image
import glob
import re

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# load the model
model = VGG16(include_top=False)
epochs = 30
margin = 40


def load_images():
    image_count = 0
    image_dict = {}
    image_list = []
    for filename in glob.glob('processed/*.jpg') + glob.glob('processed/*.bmp') + glob.glob(
            'processed/*.jpeg') + glob.glob('processed/*.png'):
        image_count += 1
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        image = model.predict(image)
        name = re.split("[\\\\\s\.]", filename)
        if name[1] in image_dict:
            image_list[image_dict[name[1]]].append(image)
        else:
            image_dict[name[1]] = len(image_list)
            image_list.append([image])
        if image_count % 100 == 0:
            print("Image count = ", image_count)
    return image_list


def partition_data(the_list, test_fraction=0.3):
    image_train = []
    image_test = []
    test_count = 0
    list_count = len(the_list)
    for l in the_list:
        if test_count / list_count < test_fraction:
            image_test.append(l)
            test_count += 1
        else:
            image_train.append(l)
    return image_train, image_test


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def image_create_pairs(the_list):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    for d in range(len(the_list)):
        for i1 in range(len(the_list[d])):
            for i2 in range(i1 + 1, len(the_list[d])):
                z1, z2 = the_list[d][i1], the_list[d][i2]
                pairs += [[z1, z2]]
                labels += [1]

        for i1 in range(len(the_list[d])):
            inc = random.randrange(1, len(the_list))
            dn = (d + inc) % len(the_list)
            i2 = random.randrange(0, len(the_list[dn]))
            z1, z2 = the_list[d][i1], the_list[dn][i2]
            pairs += [[z1, z2]]
            labels += [0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < margin
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < margin, y_true.dtype)))


# load images and partition them to train and test data
print('Loading images ...')
image_list = load_images()
random.shuffle(image_list)
image_train, image_test = partition_data(image_list)

input_shape = image_train[0][1].shape

# create train and test pairs
tr_pairs, tr_y = image_create_pairs(image_train)
te_pairs, te_y = image_create_pairs(image_test)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop(lr=0.0001)
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# you can get a clue
y_pred_tr = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred_tr)
y_pred_te = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred_te)

print('* Accuracy on training set based on a fixed threshold: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set based on a fixed threshold: %0.2f%%' % (100 * te_acc))

print('These accuracies are not important! The distances are good enough.')