import pandas as pd
import os

raw_path = "data/raw"

files = [
    "Transaction.xlsx",
    "User.xlsx",
    "City.xlsx",
    "Country.xlsx",
    "Region.xlsx",
    "Continent.xlsx",
    "Updated_Item.xlsx",
    "Type.xlsx",
    "Mode.xlsx"
]

for file in files:
    df = pd.read_excel(os.path.join(raw_path, file))
    print(f"\n📂 {file}")
    print(df.columns.tolist())