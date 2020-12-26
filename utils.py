#%%
#import talib 
import numpy as np

# Rolling function to replace pd.Series.rolling()
def rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def get_avg(closes,t):
    avg = closes[-t:].mean(axis=0)
    return avg
# 动量
def get_mom(closes,t):
    mom = (closes[-t:] / closes[-t-1:-1]).mean(axis=0)
    return mom

# 波动率
def get_vol(closes,t):
    vol = (closes[-t:] / closes[-t-1:-1]).std(axis=0)
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
    if closes.shape[0] < 2 * p:
        return np.zeros(closes.shape[1])
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
    if closes.shape[0] < fkp+skp+sdp:
        return np.zeros(closes.shape[1]),np.zeros(closes.shape[1]),np.zeros(closes.shape[1])
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
    if closes.shape[0] < p+1:
        return np.zeros(closes.shape[1])
    pct = closes[-p:] / closes[-p-1:-1] - 1
    up = pct.copy() 
    up[up<0] = np.nan 
    up = np.nanmean(up,axis=0)
    down = pct.copy() 
    down[down>0] = np.nan 
    down = -np.nanmean(down,axis=0)
    rsi = 100 - (100 / (1 + up/down))
    return rsi


# TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
def get_trix(closes,p=14):
    if closes.shape[0] < 3*p:
        return np.zeros(closes.shape[1])
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
    if closes.shape[0] < p:
        return np.zeros(closes.shape[1])
    hh = np.max(rolling_window(highs.T,p),axis=2).T # highest high
    ll = np.min(rolling_window(lows.T,p),axis=2).T # lowest low
    willr = 100 * (hh[-1] - closes[-1]) / (hh[-1] - ll[-1])
    return willr


def get_macd(closes,fp=12,sp=26):
    if closes.shape[0] < max(fp,sp):
        return np.zeros(closes.shape[1])
    w1 = np.asarray([np.power(((fp-1)/(fp+1)),fp-x-1) for x in range(fp)]) * 2 / (fp + 1)
    w2 = np.asarray([np.power(((sp-1)/(sp+1)),sp-x-1) for x in range(sp)]) * 2 / (sp + 1)
    ema1 = np.average(rolling_window(closes[-fp:].T,fp),axis=2,weights=w1)
    ema2 = np.average(rolling_window(closes[-sp:].T,sp),axis=2,weights=w2)
    macd = ema1 - ema2
    return macd


def get_natr(highs,lows,closes,p=30):
    if closes.shape[0] < p+1:
        return np.zeros(closes.shape[1])
    hl = highs[-p:] - lows[-p:]
    hcp = np.abs(highs[-p:] - closes[-p-1:-1])
    lcp = np.abs(lows[-p:] - closes[-p-1:-1])
    tr = np.maximum(hl,hcp,lcp)
    natr = np.mean(tr,axis=0) / closes[-1]
    return natr

#deprecated
def get_obv(closes,volumes,prev_factor=None):
    '''
    prev_factor: 1D-array, span the stock space
    '''
    if prev_factor is None:
        obv=np.sign(closes[-1]-closes[-2])*volumes[-1]
    else:
        obv=prev_factor+np.sign(closes[-1]-closes[-2])*volumes[-1]
    return obv

#Money Flow Index
def get_mfi(closes,highs,lows,volumes,p=14):
    if closes.shape[0] < p+1:
        return np.zeros(closes.shape[1])
    h = highs[-p-1:]
    l = lows[-p-1:]
    c = closes[-p-1:]
    tp = (h + l + c) / 3
    sign = np.sign(tp[-p:] / tp[-p-1:-1] - 1)
    up = sign.copy()#否则是指针
    up[up<0] = 0
    pos_money_flow=np.sum(up*tp[-p:]*volumes[-p:],axis=0)
    down = sign.copy()
    down[down>0] = 0
    neg_money_flow=-np.sum(down*tp[-p:]*volumes[-p:],axis=0)

    mfi=100-(100/(1+pos_money_flow/neg_money_flow))
    return mfi


# #%%
# import pandas as pd
# df=pd.read_csv('./data/CONTEST_DATA_IN_SAMPLE_1.csv',header=None)
# df.columns=['days','asset','open','high','low','close','volume']
# df=df.set_index(['days','asset'])
# df1=df.loc[[i for i in range(15)]]
# t=get_mfi(df1['close'].unstack().values,df1['high'].unstack().values,df1['low'].unstack().values,df1['volume'].unstack().values)
# t
# # %%
