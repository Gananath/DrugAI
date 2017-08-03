import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv1D, UpSampling1D, MaxPooling1D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
# time step addtition to feature


def dimX(x, ts):
    x = np.asarray(x)
    newX = []
    for i, c in enumerate(x):
        newX.append([])
        for j in range(ts):
            newX[i].append(c)
    return np.array(newX)


def train_test_split(X, y, percentage=0.75):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
    X_test = X[p:]
    Y_test = y[p:]
    return X_train, X_test, Y_train, Y_test


# time step addtition to target
def dimY(Y, ts, char_idx, chars):
    temp = np.zeros((len(Y), ts, len(chars)), dtype=np.bool)
    for i, c in enumerate(Y):
        for j, s in enumerate(c):
            # print i, j, s
            temp[i, j, char_idx[s]] = 1
    return np.array(temp)


def Discriminator(y_dash, dropout=0.4, lr=0.00001, PATH="Dis.h5"):
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
Instead, the output should be as large and negative as possible for generated inputs and as large and positive
as possible for real inputs."""
    model = Sequential()
    model.add(Conv1D(input_shape=(y_dash.shape[1], y_dash.shape[2]),
                     nb_filter=25,
                     filter_length=4,
                     border_mode='same'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D())
    model.add(Conv1D(nb_filter=10,
                     filter_length=4,
                     border_mode='same'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('linear'))

    opt = Adam(lr, beta_1=0.5, beta_2=0.9)

    #reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpoint_D = ModelCheckpoint(
        filepath=PATH, verbose=1, save_best_only=True)

    model.compile(optimizer=opt,
                  loss=wasserstein_loss,
                  metrics=['accuracy'])
    return model, checkpoint_D


def Generator(x_dash, y_dash, dropout=0.4, lr=0.00001, PATH="Gen.h5"):
    model = Sequential()
    # 6 features
    model.add(Dense(x_dash.shape[1], activation="relu", input_shape=(6,)))
    #(None, 5800)
    model.add(
        Dense(
            y_dash.shape[1] /
            4 *
            y_dash.shape[2] *
            8,
            activation="relu"))
    #(None, 29, 200)
    model.add(Reshape((y_dash.shape[1] / 4, y_dash.shape[2] * 8)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(dropout))
    model.add(UpSampling1D())
    #(None, 29, 200)
    model.add(Conv1D(y_dash.shape[2] * 8, kernel_size=4, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(dropout))
    model.add(UpSampling1D())
    #(None, 116, 50)
    model.add(Conv1D(y_dash.shape[2] * 2, kernel_size=4, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(dropout))
    #(None, 116, 25)
    model.add(Conv1D(y_dash.shape[2] * 1, kernel_size=4, padding="same"))
    model.add(Activation("softmax"))
    opt = Adam(lr, beta_1=0.5, beta_2=0.9)
    checkpoint_G = ModelCheckpoint(
        filepath=PATH, verbose=1, save_best_only=True)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
    return model, checkpoint_G

# prediction of argmax


def prediction(preds):
    y_pred = []
    for i, c in enumerate(preds):
        y_pred.append([])
        for j in c:
            y_pred[i].append(np.argmax(j))
    return np.array(y_pred)
# sequence to text conversion


def seq_txt(y_pred, idx_char):
    newY = []
    for i, c in enumerate(y_pred):
        newY.append([])
        for j in c:
            newY[i].append(idx_char[j])

    return np.array(newY)

# joined smiles output


def smiles_output(s):
    smiles = np.array([])
    for i in s:
        j = ''.join(str(k) for k in i)
        smiles = np.append(smiles, j)
    return smiles
