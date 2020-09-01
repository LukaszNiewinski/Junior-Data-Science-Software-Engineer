import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import accuracy_score

def execute(val_input_path: str, model_path: str) -> None:
    # predict predicts the target class for data in request and returns the result
    # validation should be used for training for adjsuting the parameters

    # how to preprocess validation data?
    data = pd.read_csv(val_input_path)

    y_val: np.ndarray = data.pop('Survived').values
    X_val: np.ndarray = data.values

    # read with context manager?
    model_unpickle = open(model_path, 'rb')
    model = pkl.load(model_unpickle)
    model.close()

    # keep predictions and target separated
    prdY_val = model.predict(X_val)

    accuracy_rate = accuracy_score(y_val, prdY_val)

    info = f'accuracy rate on validation set is {accuracy_rate}'
    print(info)
