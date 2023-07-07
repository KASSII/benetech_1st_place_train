import os
import re
import json
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from polyleven import levenshtein # a faster version of levenshtein

from config import CFG

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The Root Mean Square Error.
    """
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function for the given value.

    Args:
        x (float): The input value.

    Returns:
        float: The result of the sigmoid function.
    """
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the normalized Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The normalized Root Mean Square Error.
    """
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    # https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396947
    if denominator == 0:
        if numerator == 0:
            return 1.0
        return 0.0

    return sigmoid(numerator / denominator)


def normalized_levenshtein_score(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate the normalized Levenshtein distance between two lists of strings.

    Args:
        y_true (List[str]): The true values.
        y_pred (List[str]): The predicted values.

    Returns:
        float: The normalized Levenshtein distance.
    """
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(
    y_true: List[Union[float, str]], y_pred: List[Union[float, str]]
) -> float:
    """
    Calculate the score for a series of true and predicted values.

    Args:
        y_true (List[Union[float, str]]): The true values.
        y_pred (List[Union[float, str]]): The predicted values.

    Returns:
        float: The score for the series.
    """
    if "NaN" in y_true:
        y_true.remove("NaN")
    
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        y_pred = list(map(str, y_pred))
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        # Since this is a generative model, there is a chance it doesn't produce a float.
        # In that case, we return 0.0.
        try:
            return normalized_rmse(y_true, list(map(float, y_pred)))
        except:
            return 0.0


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(index=False), predictions.itertuples(index=False)
    )
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        scores.append(score_series(gt_series, pred_series))

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)["score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    return np.mean(scores), chart_type2score


def string2triplet(pred_string: str) -> Tuple[str, List[str], List[str]]:
    """
    Convert a prediction string to a triplet of chart type, x values, and y values.

    Args:
        pred_string (str): The prediction string.

    Returns:
        Tuple[str, List[str], List[str]]: A triplet of chart type, x values, and y values.
    """
    chart_type = ""

    try:
        pred_string = pred_string.split("</s>")[0]
        split_pred_string = pred_string.split(" <0x0A> ")
        chart_type_str = split_pred_string[0]
        data_series = split_pred_string[1:]

        x = []
        y = []
        nan_flag = False
        for data in data_series:
            split_data = data.split(" | ")
            if len(split_data)>=2:
                x.append(split_data[0])
                if split_data[1] != "NaN":
                    y.append(split_data[1])
                else:
                    nan_flag = True
            elif len(split_data)==1:
                x.append(split_data[0])
                y.append(0)
            
        if len(x) == 0 or len(y) == 0:
            return chart_type, [], []
        
        if not nan_flag:
            min_length = min(len(x), len(y))
            x = x[:min_length]
            y = y[:min_length]
        return chart_type, x, y
    except:
        return chart_type, [], []


def validation_metrics(val_outputs: List[str], val_ids: List[str], gt_df: pd.DataFrame, save_path=None) -> Dict[str, float]:
    """
    Calculate validation metrics for a set of outputs, ids, and ground truth dataframe.

    Args:
        val_outputs (List[str]): A list of validation outputs.
        val_ids (List[str]): A list of validation ids.
        gt_df (pd.DataFrame): The ground truth dataframe.

    Returns:
        Dict[str, float]: A dictionary containing the validation scores.
    """
    pred_triplets = []

    for example_output in val_outputs:
        pred_triplets.append(string2triplet(example_output))

    pred_df = pd.DataFrame(
        index=[f"{id_}_x" for id_ in val_ids] + [f"{id_}_y" for id_ in val_ids],
        data={
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        },
    )

    temp_df = gt_df.loc[pred_df.index.values]
    overall_score, chart_type2score = benetech_score(
        temp_df, pred_df
    )

    if save_path is not None:
        gt_ids = temp_df.index.values
        pred_ids = pred_df.index.values
        assert((gt_ids==pred_ids).all())

        gt_data_series = temp_df["data_series"].values
        pred_data_series = pred_df["data_series"].values
        gt_chart_type = temp_df["chart_type"].values
        pred_chart_type = pred_df["chart_type"].values
        score = temp_df["score"].values
        is_type_error = (gt_chart_type!=pred_chart_type)
        
        gt_data_num = np.array([len(x) for x in gt_data_series])
        pred_data_num = np.array([len(x) for x in pred_data_series])
        is_data_num_error = (gt_data_num!=pred_data_num)

        result_df = pd.DataFrame({
            "id": gt_ids,
            "score": score,
            "gt_chart_type": gt_chart_type,
            "pred_chart_type": pred_chart_type,
            "gt_data_series": gt_data_series,
            "pred_data_series": pred_data_series,
            "is_type_error": is_type_error,
            "is_data_num_error": is_data_num_error
        })
        result_df.to_csv(save_path, index=False)

    return {
        "val_score": overall_score,
        **{f"{k}_score": v for k, v in chart_type2score.items()},
    }
