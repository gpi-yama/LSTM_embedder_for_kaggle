
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from constants import *


class Preprocessing:
    def __init__(self):
        calendarDTypes = {"event_name_1": "category",
                          "event_name_2": "category",
                          "event_type_1": "category",
                          "event_type_2": "category",
                          "weekday": "category",
                          'wm_yr_wk': 'int16',
                          "wday": "int16",
                          "month": "int16",
                          "year": "int16",
                          "snap_CA": "int16",
                          'snap_TX': 'int16',
                          'snap_WI': 'int16'}

        priceDTypes = {"store_id": "category",
                       "item_id": "category",
                       "wm_yr_wk": "int16",
                       "sell_price": "float32"}

        self.calendar_df = pd.read_csv(
            data_dir+"calendar.csv", dtype=calendarDTypes)
        self.calendar_df["date"] = pd.to_datetime(self.calendar_df["date"])

        self.sales_df = pd.read_csv(
            data_dir + "sales_train_validation.csv", dtype=priceDTypes
        ).iloc[:300]
        self.price_df = pd.read_csv(data_dir+"sell_prices.csv")

    def mean_encoding_by_day(self, key):
        tmp_calendar_df = self.calendar_df.iloc[:1913]
        calendar_df["enc_"+str(key)] = 0.0
        for i, y in enumerate(tmp_calendar_df[key].unique()):
            ave = np.average(
                self.sales_df[tmp_calendar_df[tmp_calendar_df[key] == y]["d"]])
            d = tmp_calendar_df[tmp_calendar_df[key] == y].copy()
            d["enc_"+str(key)] = ave
            if i == 0:
                d_all = d
            else:
                d_all = pd.concat([d_all, d])

        tmp_calendar_df["enc_"+str(key)] = d_all["enc_"+str(key)].copy()
        tmp_calendar_df["enc_" +
                        str(key)] = tmp_calendar_df["enc_"+str(key)].fillna(0.0)

    def cat_encoding(self, key, df):
        df["cenc_"+key] = df[key].astype("category").cat.codes.astype("int16")

    def mean_encoding_by_item(self, key):
        df = self.sales_df.groupby(key).mean().mean(axis="columns").copy()
        for i, y in enumerate(self.sales_df[key].unique()):
            d = self.sales_df[self.sales_df[key] == y].copy()
            d["enc_"+key] = df[y]
            if i == 0:
                d_all = d
            else:
                d_all = pd.concat([d_all, d])
        self.sales_df["enc_"+key] = d_all["enc_"+key].copy()

    def encode_all(self):
        # # mean encoding for calendar
        # self.mean_encoding_by_day("year")
        # self.mean_encoding_by_day("event_name_1")
        # self.mean_encoding_by_day("event_name_2")
        # self.mean_encoding_by_day("weekday")
        # self.mean_encoding_by_day("month")

        # mean encoding for items
        self.mean_encoding_by_item("cat_id")
        self.mean_encoding_by_item("dept_id")
        self.mean_encoding_by_item("store_id")
        self.mean_encoding_by_item("item_id")

        self.cat_encoding("dept_id", self.sales_df)
        self.cat_encoding("item_id", self.sales_df)
        self.cat_encoding("state_id", self.sales_df)
        self.cat_encoding("store_id", self.sales_df)

        self.cat_encoding("year", self.calendar_df)
        self.cat_encoding("weekday", self.calendar_df)
        self.cat_encoding("month", self.calendar_df)
        self.cat_encoding("event_name_1", self.calendar_df)

    def standardize(self):
        # calculate standard deviation and mean of sales
        tmp_calendar_df = self.calendar_df.iloc[:1913]
        std_sales = np.std(self.sales_df[tmp_calendar_df["d"]].values)
        mean_sales = np.mean(self.sales_df[tmp_calendar_df["d"]].values)

        # standardize the sales values
        for key in tmp_calendar_df["d"].unique():
            self.sales_df[key] = self.sales_df[key].map(
                lambda x: (x - mean_sales) / std_sales)

        self.mean_sales = mean_sales
        self.std_sales = std_sales
