import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

url = "https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv"

"""
Step 1: Loading Data:
"""

df = pd.read_csv(url, skiprows=1)
# df = df.drop(df.columns[0], axis=1) # Redundant.
df = df.iloc[:, 1:].values
# df[:, [0, 1]] = df[:, [1, 0]]

"""
Step 2: Feature Scaling:
"""

sc = MinMaxScaler()
training_set = sc.fit_transform(df)


