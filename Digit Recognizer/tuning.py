import numpy as np
import pandas as pd

np.random.seed(2)

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical

def prepare_data():

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    y_train = train["label"]
    x_train = train.drop(labels = ["label"],axis=1)

    x_train = x_train / 255.0
    test = test / 255.0

    x_train = x_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)

    y_train = to_categorical(y_train,num_classes=10)
    random_seed = 2

    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.1,
                                                        random_state=random_seed)

    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    """
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()

    model.add(Conv2D(filters = 32,kernel_size = (5,5),padding="Same",activation = "relu",input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout({{uniform(0,1)}}))

    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding="Same",activation = "relu"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding="Same",activation = "relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))
    model.add(Dropout({{uniform(0,1)}}))

    model.add(Flatten())
    model.add(Dense({{choice([16,32,64,128,256,512])}},activation="relu"))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(10,activation="softmax"))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=86,
              epochs=30,
              verbose=1,
              validation_data=(x_test, y_test))

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -val_acc, 'status': STATUS_OK, 'model': model}

if __name__ == "__main__":

    best_run, best_model = optim.minimize(model=create_model,
                                          data=prepare_data,
                                          algo=tpe.suggest,
                                          max_evals=6,
                                          trials=Trials())

    print(best_model.summary())
    print(best_run)

    _, _, x_test, y_test = prepare_data()
    val_loss, val_acc = best_model.evaluate(x_test, y_test)
    print("val_loss: ", val_loss)
    print("val_acc: ", val_acc)
