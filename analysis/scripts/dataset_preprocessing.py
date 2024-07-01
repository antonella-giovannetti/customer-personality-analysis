from datetime import datetime
import sys
from typing import List, Literal, Optional

from numpy import percentile
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .dataset_analyse import find_dataframe_outliers

Period = Literal["Year", "Semester"]


def _shared_matrix_value(matrix: List[List[Optional[int]]]) -> List[Optional[int]]:
    counter_map = {}
    list_shared_values = []

    for row in matrix:
        for value in row:
            if value not in counter_map:
                counter_map[value] = 0
            else:
                counter_map[value] = 1

    for key, state in counter_map.items():
        if state == 1:
            list_shared_values.append(key)

    return list_shared_values


def binning_date_by_period(input_date: str, period: Period = "Semester") -> str:
    try:
        new_date = datetime.strptime(input_date, "%d-%m-%Y").date()
        if period == "Semester" and new_date.month >= 6:
            return new_date.replace(month=6, day=1).isoformat()
        else:
            return new_date.replace(month=1, day=1).isoformat()

    except ValueError as e:
        print(f"Not date format: {e}")
        return input_date

    except Exception as e:
        print(e)
        sys.exit(0)


def label_encode_dataframe(dataframe: DataFrame, columns: List[str]) -> DataFrame:
    label_encoder = LabelEncoder()
    for column in columns:
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
    return dataframe


def drop_outliers(
    dataframe: DataFrame, columns: List[str], percent: int = 75
) -> DataFrame:
    """
    With Tukey's method to find outliers, we change quartile by input percentile & drop outliers
    """
    for column in columns:
        if dataframe[column].dtype not in ["int64", "float64"]:
            continue
        first_quartile = percentile(dataframe[column], 100 - percent)
        third_quartile = percentile(dataframe[column], percent)
        step = 1.5 * (third_quartile - first_quartile)

        dataframe.drop(
            dataframe.loc[
                ~(
                    (dataframe[column] >= first_quartile - step)
                    & (dataframe[column] <= third_quartile + step)
                ),
                column,
            ].index,
            inplace=True,
        )
    return dataframe


def drop_shared_outliers(
    dataframe: DataFrame, columns: List[str], percent: int = 75
) -> DataFrame:

    all_outliers_indexes = find_dataframe_outliers(dataframe, columns, percent)
    shared_outliers_indexes = _shared_matrix_value(all_outliers_indexes)
    dataframe.drop([index for index in shared_outliers_indexes], inplace=True)

    return dataframe


def scale_dataframe(dataframe: DataFrame) -> DataFrame:
    scaler = StandardScaler()
    numercial_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].dtype in ("int64", "float64")
    ]
    dataframe[numercial_columns] = scaler.fit_transform(dataframe[numercial_columns])
    return dataframe
