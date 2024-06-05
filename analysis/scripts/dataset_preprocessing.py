from datetime import datetime
from typing import List, Literal

from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

Period = Literal["Year", "Semester"]


def binning_date_by_period(input_date: str, period: Period = "Semester") -> str:
    try:
        new_date = datetime.strptime(input_date, "%d-%m-%Y").date()
        if period == "Semester" and new_date.month >= 6:
            return new_date.replace(month=6, day=1).isoformat()
        else:
            return new_date.replace(month=1, day=1).isoformat()

    except:
        print("Not date format")
        return input_date


def label_encode_dataframe(dataframe: DataFrame, columns: List[str]) -> DataFrame:
    label_encoder = LabelEncoder()
    for column in columns:
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
    return dataframe

def drop_outliers(dataframe: DataFrame) -> DataFrame:
    pass