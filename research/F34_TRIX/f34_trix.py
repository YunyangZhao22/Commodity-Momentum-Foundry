"""F34 · TripleExponentialSmoothingMomentum（TRIX）最小可复现实现。

因子定义（6 段式）：
    公式    s_t = (EMA^3_t(P) - EMA^3_{t-1}(P)) / (EMA^3_{t-1}(P) + eps)
    角色    [D] Direction（方向，可独立决定多空）
    方向    s_t > 0 三重平滑趋势上行（偏多）；s_t < 0 下行（偏空）

说明：
    - 三重 EMA 用 pandas 的 ewm(span=N, adjust=False)，与因子库实现一致；
    - 本实现不依赖面板框架，输入单个品种的收盘价序列即可复现信号；
    - 中长周期（N=150~200）为独立回测验证的最优参数区间（多空 Sharpe 0.70/0.69）。

示例：
    >>> import pandas as pd
    >>> price = pd.Series([...])          # 某品种收盘价序列
    >>> sig = trix(price, lookback=100)   # 得到 F34 信号序列
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


def trix(
    price: pd.Series,
    lookback: int = 100,
) -> pd.Series:
    """计算三重指数平滑价格动量（TRIX）信号。

    Parameters
    ----------
    price : pd.Series
        单一品种的价格序列（默认用 close），index 为交易日。
    lookback : int
        三次 EMA 的 span（窗口参数 N），默认 100。

    Returns
    -------
    pd.Series
        与输入等长的 F34 信号序列；预热期（< 3*N 有效样本）为 NaN。
    """
    if not isinstance(price, pd.Series):
        price = pd.Series(price)
    if lookback <= 0:
        raise ValueError("lookback 必须为正整数")

    ema_1 = price.ewm(span=lookback, adjust=False, min_periods=lookback).mean()
    ema_2 = ema_1.ewm(span=lookback, adjust=False, min_periods=lookback).mean()
    ema_3 = ema_2.ewm(span=lookback, adjust=False, min_periods=lookback).mean()

    signal = (ema_3 - ema_3.shift(1)) / (ema_3.shift(1) + EPS)
    return signal.replace([np.inf, -np.inf], np.nan)


def trix_panel(
    close: pd.DataFrame,
    lookback: int = 100,
) -> pd.DataFrame:
    """批量计算多品种的 TRIX 信号面板。

    Parameters
    ----------
    close : pd.DataFrame
        宽表面板，index=交易日、columns=品种代码。
    lookback : int
        EMA span。

    Returns
    -------
    pd.DataFrame
        与输入同形状的信号面板。
    """
    return close.apply(lambda col: trix(col, lookback=lookback))


if __name__ == "__main__":
    # 自检：构造一段单调上升的价格，信号应为正
    demo = pd.Series(np.linspace(100.0, 130.0, 500))
    sig = trix(demo, lookback=100)
    print("F34 TRIX 信号（末尾 5 个值，单调上升序列应为正）:")
    print(sig.tail(5))
