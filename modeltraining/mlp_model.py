import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def data_split(dataset):
    label = {'side':0.0, 'facing':1.0, 'nomove':2.0}
    dataset['label'] = dataset['label'].map(label)
    X = dataset.drop('label', axis=1)
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_val = scaler.fit_transform(X_val)

    return (X_train, y_train, X_test, y_test, X_val, y_val)


def training_model(Xtrain, ytrain, Xval, yval):
    classes = np.array([0.0, 1.0, 2.0])
    y_train = np.reshape(ytrain.values, -1)
    weights = compute_class_weight('balanced', classes=classes, y=ytrain)
    d_weights = dict(enumerate(weights))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(27,), kernel_regularizer=tf.keras.regularizers.l2(0.000001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.000001)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.000001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    his = model.fit(Xtrain, ytrain, epochs=100, batch_size=32, validation_data=(Xval, yval), class_weight=d_weights)

    return model, his


def prediction_on_test(Xtest, ytest, model, history):
    loss, accuracy = model.evaluate(Xtest, ytest)
    prediction = model.predict(Xtest)

    prediction_classes = prediction.argmax(axis=1)

    print("Loss={}",loss)
    print("Accuracy={}",accuracy)

    conf = confusion_matrix(ytest, prediction_classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf, annot=True, fmt="d", ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 10))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history.get('val_accuracy', []), label='Val Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history.get('val_loss', []), label='Val Loss')
    ax2.legend()
    plt.show()


def main():
    load_dotenv()

    base = os.getenv("BASE_PATH")
    file = os.path.join(base, "all_features.csv")
    dataset = pd.read_csv(file)

    X_train, y_train, X_test, y_test , X_val, y_val = data_split(dataset)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_val.shape, y_val.shape)
    model, his = training_model(X_train, y_train, X_val, y_val)

    prediction_on_test(X_test, y_test, model, his)

    joblib.dump(model, base+"mlp_model.joblib")


if __name__=="__main__":
    main()
