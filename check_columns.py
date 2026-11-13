import pandas as pd

data = pd.read_csv("pos_signals.csv")

print("Columns in pos_signals.csv:")
print(data.columns.tolist())