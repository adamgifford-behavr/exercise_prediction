# -*- coding: utf-8 -*-
"""
This module defines the representations of the feature tables in the database. The
feature tables define the general features that make up the dataset used for model
building. Currently, the following feature tables exist in the databse:
    - naive_frequency_features
"""
# pylint:  disable=too-few-public-methods
from sqlalchemy import Column, DateTime, Float, Integer, Sequence, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import decl_api

Base: decl_api.DeclarativeMeta = declarative_base()


class NaiveFreqFeats(Base):
    """
    This class is used to define the table object for the table `naive_frequency_features`
    that exists in the database
    """

    __tablename__ = "naive_frequency_features"

    naive_frequency_features_naive_frequency_features_id_seq = Sequence(
        "naive_frequency_features_naive_frequency_features_id_seq",
        metadata=Base.metadata,
    )
    naive_frequency_features_id = Column(
        Integer,
        naive_frequency_features_naive_frequency_features_id_seq,
        server_default=(
            naive_frequency_features_naive_frequency_features_id_seq.next_value()
        ),
        primary_key=True,
    )
    featurize_id = Column(String)
    file = Column(String)
    dataset_group = Column(String)
    added_datetime = Column(DateTime)
    window_size = Column(Integer)
    t_index = Column(Integer)
    label = Column(String)
    label_group = Column(String)
    accel_x_0 = Column(Float)
    accel_x_1 = Column(Float)
    accel_x_2 = Column(Float)
    accel_x_4 = Column(Float)
    accel_x_5 = Column(Float)
    accel_x_7 = Column(Float)
    accel_x_9 = Column(Float)
    accel_x_12 = Column(Float)
    accel_x_16 = Column(Float)
    accel_y_0 = Column(Float)
    accel_y_1 = Column(Float)
    accel_y_2 = Column(Float)
    accel_y_3 = Column(Float)
    accel_y_4 = Column(Float)
    accel_y_5 = Column(Float)
    accel_y_6 = Column(Float)
    accel_y_7 = Column(Float)
    accel_y_8 = Column(Float)
    accel_y_9 = Column(Float)
    accel_y_10 = Column(Float)
    accel_y_11 = Column(Float)
    accel_y_12 = Column(Float)
    accel_z_0 = Column(Float)
    accel_z_1 = Column(Float)
    accel_z_2 = Column(Float)
    accel_z_3 = Column(Float)
    accel_z_4 = Column(Float)
    accel_z_5 = Column(Float)
    accel_z_8 = Column(Float)
    accel_z_12 = Column(Float)
    accel_z_14 = Column(Float)
    gyro_x_0 = Column(Float)
    gyro_x_1 = Column(Float)
    gyro_x_2 = Column(Float)
    gyro_x_3 = Column(Float)
    gyro_x_4 = Column(Float)
    gyro_x_5 = Column(Float)
    gyro_x_6 = Column(Float)
    gyro_x_7 = Column(Float)
    gyro_x_8 = Column(Float)
    gyro_x_9 = Column(Float)
    gyro_x_10 = Column(Float)
    gyro_x_11 = Column(Float)
    gyro_x_12 = Column(Float)
    gyro_x_13 = Column(Float)
    gyro_y_0 = Column(Float)
    gyro_y_1 = Column(Float)
    gyro_y_2 = Column(Float)
    gyro_y_3 = Column(Float)
    gyro_y_4 = Column(Float)
    gyro_y_5 = Column(Float)
    gyro_y_6 = Column(Float)
    gyro_y_7 = Column(Float)
    gyro_y_8 = Column(Float)
    gyro_y_10 = Column(Float)
    gyro_y_11 = Column(Float)
    gyro_z_0 = Column(Float)
    gyro_z_1 = Column(Float)
    gyro_z_3 = Column(Float)
    gyro_z_4 = Column(Float)
    gyro_z_5 = Column(Float)
    gyro_z_8 = Column(Float)
    gyro_z_14 = Column(Float)
    gyro_z_17 = Column(Float)
    gyro_z_18 = Column(Float)

    def __repr__(self):
        return (
            f"<NaiveFreqFeats(featurize_id='{self.featurize_id}', file='{self.file}', "
            f"dataset_group={self.dataset_group}, added_datetime={self.added_datetime}), "
            f"window_size={self.window_size},  t_index={self.t_index}>"
        )
