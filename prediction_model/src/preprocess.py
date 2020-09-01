import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from typing import Tuple


def inject_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Injects missing values, median for numeric and most frequent for category."""
    sb_vars = data.select_dtypes(include='object')
    data[sb_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

    cols_nr, cols_sb = get_subset_by_type(data)

    imp_nr = SimpleImputer(strategy='median', missing_values=np.nan, copy=True)
    imp_sb = SimpleImputer(strategy='most_frequent', missing_values=np.nan, copy=True)
    df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
    df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

    data = df_nr.join(df_sb, how='right')

    return data

def min_max_norm(data: pd.DataFrame) -> pd.DataFrame:
    """Applies min_max normalization."""

    cols_nr, cols_sb = get_subset_by_type(data)
    df_nr = data[cols_nr]

    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    df_nr = pd.DataFrame(transf.transform(df_nr), columns=df_nr.columns)
    norm_data_minmax = df_nr.join(data[cols_sb], how='right')

    return norm_data_minmax

def get_subset_by_type(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get dataframe subsets of numeric and category type"""
    cols_nr = data.select_dtypes(include='number')
    cols_sb = data.select_dtypes(include='category')

    return cols_nr, cols_sb

def execute(input_path: str, output_path: str) -> None:
    """Preprocess - missing values injection, column drop, normalization"""
    data = pd.read_csv(input_path, sep = ";", index_col='PassengerId')
    # drop 'Name' as each records has unique value and it's not informative
    data.drop('Name', axis=1, inplace=True)

    #Drop 'Cabin' as it has too many missing records
    data.drop('Cabin', axis=1, inplace=True)

    data = inject_missing_values(data)
    data = min_max_norm(data)

    # drop remaining missing records
    data = data.dropna()

    data.to_csv(output_path)
