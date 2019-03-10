import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def load_dataset():
    iris_dataset = load_iris()
    X = iris_dataset['data']
    y = iris_dataset['target']
    return X, y

def construct_model():
    model = Sequential()
    model.add(Dense(16, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

X, y = load_dataset()
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = construct_model()
optimizer = SGD(lr=0.05)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(X_train, y_train,  batch_size=5, epochs=100)

print("Model performance...")
y_ = model.predict(X_test)

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_, axis=1)
print("Confusion matrix:\n{}".format(confusion_matrix(y_true , y_pred)))