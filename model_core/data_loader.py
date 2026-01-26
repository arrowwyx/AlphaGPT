import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        
    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        top_query = f"""
        SELECT address FROM tokens 
        LIMIT {limit_tokens} 
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not addrs: raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(addrs) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.fillna(method='ffill').fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")

    def load_data_from_csv(
        self,
        csv_path,
        address="BTCUSDT",
        time_col="time",
        default_liquidity=None,
        default_fdv=None
    ):
        print("Loading data from CSV...")
        df = pd.read_csv(csv_path)

        if time_col not in df.columns:
            for alt in ["timestamp", "date", "datetime", "time"]:
                if alt in df.columns:
                    time_col = alt
                    break
            else:
                raise ValueError(f"Missing time column. Tried '{time_col}'.")

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "liquidity" not in df.columns:
            if default_liquidity is None:
                raise ValueError("Missing 'liquidity' column and no default provided.")
            df["liquidity"] = float(default_liquidity)

        if "fdv" not in df.columns:
            if default_fdv is None:
                raise ValueError("Missing 'fdv' column and no default provided.")
            df["fdv"] = float(default_fdv)

        df = df.copy()
        df["address"] = address
        df = df.rename(columns={time_col: "time"})
        df = df.sort_values("time")

        def to_tensor(col):
            pivot = df.pivot(index="time", columns="address", values=col)
            pivot = pivot.fillna(method="ffill").fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            "open": to_tensor("open"),
            "high": to_tensor("high"),
            "low": to_tensor("low"),
            "close": to_tensor("close"),
            "volume": to_tensor("volume"),
            "liquidity": to_tensor("liquidity"),
            "fdv": to_tensor("fdv")
        }
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache["open"]
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")