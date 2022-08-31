# -*- coding: utf-8 -*-
"""
This module defines the representations of the predictions tables in the database. The
predictions tables record the predictions of a given model version associated with the
features used from the specified features table. Currently, the following predictions
tables exist in the databse:
    - naive_frequency_features_predictions (links to features table
    naive_frequency_features)
"""
# pylint:  disable=too-few-public-methods
# pylint: disable=line-too-long

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, Sequence, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import decl_api

from src.features.features_tables import NaiveFreqFeats

Base: decl_api.DeclarativeMeta = declarative_base()


class NaiveFreqFeatsPred(Base):
    """
    This class is used to define the table object for the table
    `naive_frequency_features_predictions` that exists in the database
    """

    __tablename__ = "naive_frequency_features_predictions"

    naive_frequency_features_predictions_naive_frequency_features_predictions_id_seq = (
        Sequence(
            (
                "naive_frequency_features_predictions_naive_frequency_features_"
                "predictions_id_seq"
            ),
            metadata=Base.metadata,
        )
    )
    naive_frequency_features_predictions_id = Column(
        Integer,
        naive_frequency_features_predictions_naive_frequency_features_predictions_id_seq,
        server_default=(
            naive_frequency_features_predictions_naive_frequency_features_predictions_id_seq.next_value()
        ),
        primary_key=True,
    )
    naive_frequency_features_id = Column(
        Integer, ForeignKey(NaiveFreqFeats.naive_frequency_features_id)
    )
    featurize_id = Column(String)
    file = Column(String)
    window_size = Column(Integer)
    t_index = Column(Integer)
    true_label = Column(String)
    predicted_label = Column(String)
    prediction_correct = Column(Boolean)
    model_version = Column(String)
    added_datetime = Column(DateTime)
