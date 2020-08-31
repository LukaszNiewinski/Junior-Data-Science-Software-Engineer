import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from typing import Tuple, List

def decompose_ticket(value: str) -> Tuple[str, int]:
    """Auxiliary function to decompose Ticket and retrieve its str prefix and int ticket number."""
    ticket = value.rsplit(" ", 1)
    ticket_number = None

    # ticket can be composed of its prefix and number
    if len(ticket) > 1:
        prefix = ticket[0]
        ticket_number = int(ticket[1])
    elif ticket[0].isdigit():
        prefix = 'no_prefix'
        ticket_number = int(ticket[0])
    # there might be a ticket that does not contain a number
    else:
        prefix = ticket[0]

    return (prefix, ticket_number)

def build_ticket_features(data: pd.DataFrame) -> pd.DataFrame:
    """Function to build features out of Ticket - ticket prefix and ticket number"""
    COLUMNS = ['Ticket_prefix', 'Ticket_number']
    df = pd.DataFrame(data['Ticket'].map(decompose_ticket).values.tolist(), index=data.index, columns=COLUMNS)
    data = data.join(df, how='right')
    return data

def dummify(data: pd.DataFrame, cols_to_dummify: List[str]) -> pd.DataFrame:
    """Function to dummify features of category type"""
    one_hot_encoder = OneHotEncoder(sparse=False)

    for var in cols_to_dummify:
        one_hot_encoder.fit(data[var].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([var])
        transformed_data = one_hot_encoder.transform(data[var].values.reshape(-1, 1))
        data = pd.concat((data, pd.DataFrame(transformed_data, columns=feature_names)), 1)
        data.pop(var)
    return data

def execute(input_path: str, output_path: str) -> None:
    """Builds features"""
    data = pd.read_csv(input_path, index_col='PassengerId', sep=';', decimal='.')

    data = build_ticket_features(data)

    # change type of a 'object' type features
    sb_vars = data.select_dtypes(include='object')
    data[sb_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    cols_sb = data.select_dtypes(include='category')

    # create separate columns for embark, gender and ticket prefix built column
    df = dummify(data, cols_sb.columns)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df.to_csv(output_path, sep=';', decimal='.')
