"""
Smoke test for read_frame normalization on sample agent_data files.
"""

from pathlib import Path
from threadx.data.io import read_frame

SAMPLES = [
    Path(r"D:/agent_data/PLUMEUSDC_1h.parquet"),
    Path(r"D:/agent_data/1000CATUSDC_1h.json"),
]

for p in SAMPLES:
    print("---", p)
    try:
        df = read_frame(p, normalize=True)
        print("shape:", df.shape)
        print("columns:", df.columns.tolist())
        print(df.head(2))
    except Exception as e:
        print("ERROR:", type(e).__name__, e)
