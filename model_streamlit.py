import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, classification_report

def preprocess_data(raw_data):
    return pd.get_dummies(raw_data, columns=['Gender'])

def train_model(X, y):
    clf = SVC(kernel="linear", probability=True, random_state=0)
    clf.fit(X, y)
    return clf

def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    pred_proba = clf.predict_proba(X)
    accuracy = accuracy_score(y, y_pred)
    log_loss_value = log_loss(y, pred_proba)
    precision = precision_score(y, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=1)
    report = classification_report(y, y_pred, zero_division=1)
    return accuracy, log_loss_value, precision, recall, f1, report

def predict(clf, X_test):
    return clf.predict(X_test)