#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


def generate_model():
    # Model generated with optimised parameters (see .ipynb file)
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=21,
        max_features=3,
        min_samples_leaf=2,
        min_samples_split=5,
    )

    # recover the training data from the data file

    utils_path = Path(__file__).resolve().parent
    ml_data = utils_path.parent / "ml_data" / "ml_training_data_13_01_2020.zip"

    training_df = pd.read_json(ml_data, compression="zip")

    training_set, training_labels = (
        training_df.drop("DRUGGABLE", axis=1),
        training_df["DRUGGABLE"],
    )

    return rf_model.fit(training_set, training_labels)


def predict(model, data):
    df = data.copy()
    df.index = df.Target_id
    df.drop(columns=["Target_id"], inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)
    df = df.fillna(0)
    # Only columns to consider in the model (see .ipynb file for selection of the columns)
    col_to_drop = [
        "OT_max_association_score",
        "Heart_alert",
        "Liver_alert",
        "Kidney_alert",
        "dis_AScore",
        "bio_EScore",
        "safe_EScore",
        "chembl_selective_M",
        "chembl_selective_G",
        "chembl_selective_E",
        "bindingDB_phase2",
        "commercial_potent",
        "information_score",
        "gen_AQualScore",
        "genetic_NORM",
    ]
    df = df.drop(columns=col_to_drop, axis=1)

    return model.predict(df)


def predict_prob(model, data):
    df = data.copy()
    df.index = df.Target_id
    df.drop(columns=["Target_id"], inplace=True)
    df.replace({True: 1, False: 0}, inplace=True)
    df = df.fillna(0)
    # Only columns to consider in the model (see .ipynb file for selection of the columns
    col_to_drop = [
        "OT_max_association_score",
        "Heart_alert",
        "Liver_alert",
        "Kidney_alert",
        "dis_AScore",
        "bio_EScore",
        "safe_EScore",
        "chembl_selective_M",
        "chembl_selective_G",
        "chembl_selective_E",
        "bindingDB_phase2",
        "commercial_potent",
        "information_score",
        "gen_AQualScore",
        "genetic_NORM",
    ]
    df = df.drop(columns=col_to_drop, axis=1)
    return model.predict_proba(df)


def in_training_set(data):
    utils_path = Path(__file__).resolve().parent
    ml_data = utils_path.parent / "ml_data" / "ml_training_data_13_01_2020.zip"

    training_df = pd.read_json(ml_data, compression="zip")

    df = data.copy()
    df.index = df.Target_id

    df["Is_in_training_set"] = "No"
    df.loc[df.index.isin(training_df.index), ["Is_in_training_set"]] = "Yes"
    return df["Is_in_training_set"].values
