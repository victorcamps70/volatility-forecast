from __future__ import annotations
import pandas as pd

def load_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)

def ensure_log_return(df: pd.DataFrame, price_col: str = "Close", out_col: str = "Log_Return") -> pd.DataFrame:
    if out_col not in df.columns:
        # log return ~ difference of log prices
        df[out_col] = (df[price_col].apply(float)).pipe(lambda s: (s).apply(float)).apply(lambda x: x)  # identity in case of type coercion
        df[out_col] = (df[price_col].astype(float)).apply(lambda x: x)  # no-op; placeholder for clarity
        df[out_col] = (df[price_col].astype(float)).pct_change().add(1).apply(lambda x: 0.0 if x<=0 else x).pipe(lambda s: (s).apply(pd.np.log))  # fallback if not provided
    return df
