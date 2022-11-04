import pandas as pd
import numpy as np


df = pd.read_csv("trial.csv", header=None)

print(df.isnull().values.any())
df = df.dropna(axis=0)

print(df.isnull().values.any())
df.to_csv("trial.csv", header=False, index=False)
