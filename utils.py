#import talib 
import numpy as np

# Rolling function to replace pd.Series.rolling()
def rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

# 动量
def get_mom(closes,t):
    mom = closes[-t:].mean(axis=0)
    return mom

# 波动率
def get_vol(closes,t):
    vol = closes[-t:].std(axis=0)
    return vol

# 52周最高
def get_52weekhigh(closes):
    high=closes[-252:].max(axis=0)
    return (closes[-1]-high)/high

# 52周最低
def get_52weeklow(closes):
    low=closes[-252:].min(axis=0)
    return (closes[-1]-low)/low

# 价量相关性


# 下行波动占比


# CCI Commodity Channel Index
def get_cci(highs,lows,closes,p=14):
    '''
    Args:
        highs: 2D-array of high prices, time as row indexes and each stock for one column
        lows: 2D-array of low prices, time as row indexes and each stock for one column
        closes: 2D-array of close prices, time as row indexes and each stock for one column
        p: time period
    Return:
        cci: Commodity Channel Index of all stocks as a 1D-array
    '''
    h = highs[-2*p:]
    l = lows[-2*p:]
    c = closes[-2*p:]

    tp = (h + l + c) / 3

    ret = np.cumsum(tp, axis=0)
    ret[p:] = ret[p:] - ret[:-p]
    ma = ret[p:] / p
    md = np.mean(tp[p:]-ma,axis=0)
    cci = (tp[-1] - ma[-1]) / (0.015 * md)
    return cci


# KDJ (Stochastic)
def get_kdj(highs,lows,closes,fkp=9,skp=3,sdp=3):
    '''
    Args:
        highs: 2D-array of high prices, time as row indexes and each stock for one column
        lows: 2D-array of low prices, time as row indexes and each stock for one column
        closes: 2D-array of close prices, time as row indexes and each stock for one column
        fkp: fast k period, time period to calculate rsv
        skp: slow k period, time period to calculate k
        sdp: slow d period, time period to calculate d
    Return:
        cci: Commodity Channel Index of all stocks as a 1D-array
    '''
    hh = np.max(rolling_window(highs.T,fkp),axis=2).T # highest high
    ll = np.min(rolling_window(lows.T,fkp),axis=2).T # lowest low
    rsv = 100 * (closes[-hh.shape[0]:] - ll) / (hh - ll)
    K = np.mean(rolling_window(rsv.T,skp),axis=2).T
    D = np.mean(rolling_window(K.T,sdp),axis=2).T[-1]
    K = K[-1]
    J = 3 * D - 2 * K
    return K,D,J


# RSI Relative Strength Index
def get_rsi(closes,p=14):
    '''
    Args:
        closes: 2D-array of close prices, time as row indexes and each stock for one column
        p: time period
    Return:
        cci: Commodity Channel Index of all stocks as a 1D-array
    '''
    p = 14
    pct = closes[-p:] / closes[-p-1:-1] - 1
    up = pct 
    up[up<0] = np.nan 
    up = np.nanmean(up,axis=0)
    down = pct 
    down[down>0] = np.nan 
    down = np.nanmean(down,axis=0)
    rsi = 100 - (100 / (1 + up/down))
    return rsi


# TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
def get_trix(closes,p=14):
    w = np.asarray([np.power(((p-1)/(p+1)),p-x-1) for x in range(p)]) * 2 / (p + 1)
    ema1 = np.average(rolling_window(closes[-3*p:].T,p),axis=2,weights=w)
    ema2 = np.average(rolling_window(ema1,p),axis=2,weights=w)
    ema3 = np.average(rolling_window(ema2,p),axis=2,weights=w).T
    trix = 100 * (ema3[-1] / ema3[-2] - 1)
    return trix


def get_willr(highs,lows,closes,p=14):
    '''
    Args:
        highs: 2D-array of high prices, time as row indexes and each stock for one column
        lows: 2D-array of low prices, time as row indexes and each stock for one column
        closes: 2D-array of close prices, time as row indexes and each stock for one column
        p: time period
    Return:
        willr: Williams' %R of all stocks as a 1D-array
    '''
    hh = np.max(rolling_window(highs.T,p),axis=2).T # highest high
    ll = np.min(rolling_window(lows.T,p),axis=2).T # lowest low
    willr = 100 * (hh[-1] - closes[-1]) / (hh[-1] - ll[-1])
    return willr


def get_macd(closes,fp=12,sp=26):
    w1 = np.asarray([np.power(((fp-1)/(fp+1)),fp-x-1) for x in range(fp)]) * 2 / (fp + 1)
    w2 = np.asarray([np.power(((sp-1)/(sp+1)),sp-x-1) for x in range(sp)]) * 2 / (sp + 1)
    ema1 = np.average(rolling_window(closes[-fp:].T,fp),axis=2,weights=w1)
    ema2 = np.average(rolling_window(closes[-sp:].T,sp),axis=2,weights=w2)
    macd = ema1 - ema2
    return macd


def get_natr(highs,lows,closes,p):
    p = 30
    hl = highs[-p:] - lows[-p:]
    hcp = np.abs(highs[-p:] - closes[-p-1:-1])
    lcp = np.abs(lows[-p:] - closes[-p-1:-1])
    tr = np.maximum(hl,hcp,lcp)
    natr = np.mean(tr,axis=0) / closes[-1]
    return natr


