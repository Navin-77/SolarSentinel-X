from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent

df = pd.read_csv(
    project_root / "dataset" / "synthetic" / "solar_microgrid_dataset.csv"
)

print(df.tail(1).T)