import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from dotenv import load_dotenv
import joblib
from mlp_model import data_split
import matplotlib.pyplot as plt
import seaborn as sns


def training_model(X_train, y_train):
    params_grid = {'kernel': ['rbf'], 'C': [800, 1000, 1500, 2000], 'gamma': [0.001, 0.05, 0.1, 0.5]}

    model = SVC()

    model_cv = GridSearchCV(model, param_grid=params_grid, cv=5, verbose=1)

    print('########################')
    model_cv.fit(X_train, y_train)

    best_params = model_cv.best_params_
    best_estimator = model_cv.best_estimator_

    return best_estimator, best_params


def predicting(X, y, model):
    y_pred = model.predict(X)

    acc_score = accuracy_score(y, y_pred)
    fone_score = f1_score(y, y_pred, average='weighted')
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')

    conf = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf, annot=True, fmt="d", ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

    return {'accuracy':acc_score, 'f1_score':fone_score, 'precision':precision, 'recall':recall}


def main():
    load_dotenv()

    base = os.getenv("BASE_PATH")
    file = os.path.join(base, "all_features.csv")
    dataset = pd.read_csv(file)

    X_train, y_train, X_test, y_test, X_val, y_val = data_split(dataset)

    model, params = training_model(X_train, y_train)

    joblib.dump(model, base + 'multi_class_svm.joblib')

    score = predicting(X_test, y_test, model)
    print("Prediction on test dataset: {}".format(score))
    print("Best parameters for SVM: {}".format(params))


if __name__=="__main__":
    main()
