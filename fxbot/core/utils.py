import importlib
import numpy as np
import pandas as pd

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def adx(high, low, close, n=14):
    up = high.diff(); dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    pdi = 100*(pd.Series(plus_dm,index=high.index).rolling(n).mean()/atr)
    mdi = 100*(pd.Series(minus_dm,index=high.index).rolling(n).mean()/atr)
    dx  = 100*(pdi-mdi).abs()/(pdi+mdi)
    return dx.rolling(n).mean()

def atr(high, low, close, n=14):
    tr1=(high-low).abs()
    tr2=(high-close.shift()).abs()
    tr3=(low -close.shift()).abs()
    tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def donchian(high, low, n=20):
    return high.rolling(n).max(), low.rolling(n).min()

def import_from_path(path: str):
    mod, cls = path.rsplit(".",1)
    return getattr(importlib.import_module(mod), cls)
