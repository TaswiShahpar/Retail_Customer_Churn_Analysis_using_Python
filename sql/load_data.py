import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import os
connection_url=URL.create(
    drivername = "mysql+pymysql",
    username="root",
    password = "Capsule@1603",   
    host     = "127.0.0.1",
    port     = "3306",
    database    = "churn_analytics"
    )


# --- Paths ---
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")

TRANSACTIONS_CSV = os.path.join(PROCESSED, "online_retail_cleaned.csv")
FEATURES_CSV     = os.path.join(PROCESSED, "customer_churn_features.csv")
PREDICTIONS_CSV  = os.path.join(PROCESSED, "customer_churn_predictions.csv")

def load_data():
    engine = create_engine(
        connection_url
    )
    print("MySQL connected")

    print("\nReading CSV files")
    df_tx   = pd.read_csv(TRANSACTIONS_CSV, encoding="utf-8")
    df_feat = pd.read_csv(FEATURES_CSV,     encoding="utf-8")
    df_pred = pd.read_csv(PREDICTIONS_CSV,  encoding="utf-8")

    
    price_col = "Price" if "Price" in df_tx.columns else "UnitPrice"
    df_tx["Revenue"] = df_tx["Quantity"] * df_tx[price_col]

    print(f"  transactions       : {len(df_tx):,} rows | columns: {list(df_tx.columns)}")
    print(f"  customer_features  : {len(df_feat):,} rows | columns: {list(df_feat.columns)}")
    print(f"  customer_predictions:{len(df_pred):,} rows | columns: {list(df_pred.columns)}")

    df_tx.to_sql("transactions",           engine, if_exists="replace", index=False)
    print("transactions table done")

    df_feat.to_sql("customer_features",    engine, if_exists="replace", index=False)
    print("customer_features table done")

    df_pred.to_sql("customer_predictions", engine, if_exists="replace", index=False)
    print("customer_predictions table done")

    print("\nDone! churn_analytics database is ready in MySQL.")
if __name__ == "__main__":
    load_data()