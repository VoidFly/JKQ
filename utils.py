import talib 
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
def get_cci(highs,lows,closes,p):
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



# def get_factors(data):
#     '''
#     @params:
#     data: 2D-array, rows as daily numbers, columns as open,high,low,close,volume respectively

#     @returns:
#     factors: 1D-array
#     '''
#     o = data[:,0]
#     h = data[:,1]
#     l = data[:,2]
#     c = data[:,3]
#     v = data[:,4]
#     mom = talib.MOM(c,timeperiod=10)
#     vol = talib.STDDEV(c,timeperiod=10)
#     max52 = talib.MAX(h,timeperiod=252)
#     min52 = talib.MIN(l,timeperiod=252)

#     # 价量相关性、下行波动占比，ij是什么？
#     # 
#     # clr =  # close_volume relation
#     # dvr =  # downstream volatility ratio

#     # Commodity Channel Index
#     cci = talib.CCI(h,l,c,timeperiod=14)

#     # KDJ (Stochastic)
#     K, D = talib.STOCH(h,l,c,fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
#     J = 3 * D - 2 * K

#     # Relative Strength Index
#     rsi = talib.RSI(c,timeperiod=14)

#     # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
#     trix = talib.TRIX(c, timeperiod=30)

#     # Williams' %R
#     willr = talib.WILLR(h, l, c, timeperiod=14)

#     # Moving Average Convergence/Divergence
#     macd, macdsignal, macdhist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)

#     # Normalized Average True Range
#     atr = talib.NATR(h,l,c,timeperiod=14)

#     return np.asarray([mom, vol, max52, min52, cci, K, D, J, rsi, trix, willr, macd, macdsignal, macdhist, atr]).transpose()


