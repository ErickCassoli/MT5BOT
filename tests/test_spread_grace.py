import os
import sys
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'fxbot'))
from fxbot.exec.execution import spread_grace


def test_spread_grace_true(monkeypatch):
    df_e = pd.DataFrame({'h':[1,1], 'l':[0,0], 'c':[0.9,1.0]})
    df_r = pd.DataFrame({'h':[1,1], 'l':[0,0], 'c':[0.9,1.0]})
    params = {'donchian':14, 'adx_thr':14, 'near_by_atr_ratio':0.5}

    monkeypatch.setattr('fxbot.exec.execution.adx', lambda h,l,c,n=14: pd.Series([0,20]))
    monkeypatch.setattr('fxbot.exec.execution.donchian', lambda h,l,n: (pd.Series([1,1]), pd.Series([0,0])))

    assert spread_grace(df_e, df_r, 1.0, 0.9, params)


def test_spread_grace_false_when_spread_alto(monkeypatch):
    df_e = pd.DataFrame({'h':[1,1], 'l':[0,0], 'c':[0.9,1.0]})
    df_r = pd.DataFrame({'h':[1,1], 'l':[0,0], 'c':[0.9,1.0]})
    params = {'donchian':14, 'adx_thr':14, 'near_by_atr_ratio':0.5}

    monkeypatch.setattr('fxbot.exec.execution.adx', lambda h,l,c,n=14: pd.Series([0,20]))
    monkeypatch.setattr('fxbot.exec.execution.donchian', lambda h,l,n: (pd.Series([1,1]), pd.Series([0,0])))

    assert not spread_grace(df_e, df_r, 1.0, 2.0, params)
