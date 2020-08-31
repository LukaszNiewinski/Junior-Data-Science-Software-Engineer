import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def execute(input_path: str, model_path: str) -> None:
    """Trains model on train data and saves model."""

    # Split the data for training.
    data = pd.read_csv(input_path)

    y: np.ndarray = data.pop('Survived').values
    X: np.ndarray = data.values

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.8, stratify=y)

    # Create a classifier and select scoring methods.
    clf = RandomForestClassifier(n_estimators=10)

    # Fit full model and predict on both train and test.
    clf.fit(trnX, trnY)
    prdY = clf.predict(trnY)
    model_name = clf.__class__
    model_accuracy = accuracy_score(y, prdY)

    model_pickle = open(model_path, 'wb')
    pkl.dump(clf, model_pickle)
    model_pickle.close()

    # Return metrics and model name.
    info = f'train accuracy for {model_name} is {model_accuracy}'
    print(info)
