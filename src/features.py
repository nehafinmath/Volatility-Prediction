import numpy as np
import pandas as pd

def log_return(series):
    return np.log(series).diff()

def realized_volatility(log_returns):
    return np.sqrt(np.sum(log_returns**2))

def order_book_data_feature_construction(path):
    df = pd.read_parquet(path)

    # spreads
    df["ask_spread"] = df["ask_size1"] - df["ask_size2"]
    df["bid_spread"] = df["bid_size1"] - df["bid_size2"]

    df["total_volume"] = (
        df["ask_size1"] + df["ask_size2"] +
        df["bid_size1"] + df["bid_size2"]
    )

    df["volume_imbalance"] = abs(
        df["ask_size1"] + df["ask_size2"]
        - df["bid_size1"] - df["bid_size2"]
    )

    # WAP
    df["wap1"] = (
        df["bid_price1"] * df["ask_size1"] +
        df["ask_price1"] * df["bid_size1"]
    ) / (df["ask_size1"] + df["bid_size1"])

    df["wap2"] = (
        df["bid_price2"] * df["ask_size2"] +
        df["ask_price2"] * df["bid_size2"]
    ) / (df["ask_size2"] + df["bid_size2"])

    df["price_spread"] = (df["ask_price1"] / df["bid_price1"]) - 1
    df["wap_balance"] = df["wap1"] - df["wap2"]

    # log returns
    df["log_return1"] = df.groupby("time_id")["wap1"].apply(log_return)
    df["log_return2"] = df.groupby("time_id")["wap2"].apply(log_return)

    df = df.dropna(subset=["log_return1"])

    return df
