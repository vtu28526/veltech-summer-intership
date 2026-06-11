import os
import sys
from typing import List, Tuple


def read_dataset(path: str):
    """Read CSV or Excel into a pandas DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required. Install with: pip install pandas openpyxl") from e

    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {ext}. Use .csv or .xlsx/.xls")


def ensure_numeric_df(df):
    import pandas as pd

    num_df = df.copy()
    for c in num_df.columns:
        # Try convert only if dtype is object-like
        if num_df[c].dtype == object:
            # Convert strings like '12,3' not handled; keep simple
            num_df[c] = pd.to_numeric(num_df[c], errors="ignore")
    return num_df.select_dtypes(include="number")


def ask_path() -> str:
    path = input("Enter path to CSV/XLSX dataset: ").strip().strip('"').strip("'")
    if not path:
        raise SystemExit("No path provided.")
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    return path


def print_header(title: str):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def main_dataset_arg(argv: List[str]) -> Tuple[str, str]:
    """Return (path, name)."""
    if len(argv) >= 2:
        path = argv[1]
    else:
        path = ask_path()
    name = os.path.basename(path)
    return path, name

