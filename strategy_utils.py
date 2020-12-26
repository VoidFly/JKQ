import pandas as pd
import numpy as np
from utils import *

def get_factors(data,prev_factors):
    '''
    1*stock
    prev_factors
    '''
    if not prev_factors.empty:
        prev_factors=prev_factors.set_index(['day','stock'])

    o = data['open'].unstack(level=1)
    stock=o.columns.values
    day=o.index[-1]
    o=o.values
    h = data['high'].unstack(level=1).values
    l = data['low'].unstack(level=1).values
    c = data['close'].unstack(level=1).values
    v = data['volume'].unstack(level=1).values

    avg=get_avg(c,3)
    mom=get_mom(c,5)
    vol=get_vol(c,5)
    
    max52=get_52weekhigh(c)
    #min52=get_52weeklow(c)
    cci = get_cci(h,l,c)
    K,D,J = get_kdj(h,l,c)
    rsi = get_rsi(c)
    trix = get_trix(c)
    willr = get_willr(h,l,c)
    macd = get_macd(c)
    natr = get_natr(h,l,c)
    # print(natr)
    mfi=get_mfi(c,h,l,v)
    # print(mfi)
    #mfi可能更好
    # if prev_factors.empty:
    #     obv=get_obv(c,v)
    # else:
    #     obv=get_obv(c,v,prev_factors['obv'].unstack(level=1).values[0])

    result=pd.DataFrame([avg,mom, vol, max52, cci, K, D, J, rsi, trix, willr, macd, natr,mfi],
                index=['avg','mom', 'vol', 'max52', 'cci', 'K', 'D', 'J', 'rsi', 'trix', 'willr', 'macd', 'natr','mfi'],dtype=float).T

    result['day']=day
    result['stock']=stock
    result['close']=c[-1]
    return result # stocks|day|factors....|close

def select_factors(factors,n=10,period=5):
    factors=factors.set_index(['day','stock'])
    factor_names=factors.columns.drop(['close'])
    
    c=factors['close'].unstack(level=1).pct_change(period).shift(-period) 
    def spearman(x):
        spear=x.unstack().rank(axis=1)-c.rank(axis=1)
        spearcorrs=6*(spear*spear).sum(axis=1)/351/(351*351-1)
        return spearcorrs.mean()
    ic=factors[factor_names].apply(lambda x:spearman(x))#lambda x:x.unstack(),axis=1)
    factor_select=ic.nlargest(n).index.to_list()
    
    return factor_select

# def get_weight(factors,n=10,max_exposure=0.1,index_direction='neutral'):

#     factors=factors.set_index('stock')
#     head=[]
#     tail=[]
    
#     #简单取并集/或交集
#     for col in factors.columns:
#         head.extend(factors[col].nlargest(n).index.tolist())
#         tail.extend(factors[col].nsmallest(n).index.tolist())
#     head=list(set(head))
#     tail=list(set(tail))

#     intersect=np.intersect1d(head,tail)
#     head=np.setdiff1d(head,intersect)
#     tail=np.setdiff1d(tail,intersect)
#     #print('head num',len(head))
#     #print('tail num',len(tail))

#     #根据大盘调整仓位占比
#     weight=pd.Series(0,index=factors.index)
#     if index_direction=='up':
#         weight[head]=((1+max_exposure)/2)/len(head)
#         weight[tail]=-((1-max_exposure)/2)/len(tail)
#     elif index_direction=='down':
#         weight[head]=((1-max_exposure)/2)/len(head)
#         weight[tail]=-((1+max_exposure)/2)/len(tail)
#     else:
#         weight[head]=0.5/(len(head))
#         weight[tail]=-0.5/(len(tail))

#     #print('weight',sum(weight))
#     return weight

def get_position(weights,dailystk,prev_pos,
                prev_capital,comission):
    '''
    target_pos-> list
    TODO comission prev_pos check
    
    '''
    close=dailystk.set_index(['stock'])['close']

    target_pos=prev_capital*weights/close
    target_pos=target_pos.astype(int)
    print(target_pos[target_pos!=0])
    return target_pos.tolist()

def dnn_weight(factors,model,n=10,max_exposure=0.09):
    factors=factors.set_index('stock')
    # pred = model.pred(factors.values)
    pred = pd.Series(model.predict(factors.values).flatten())
    head = pred.nlargest(n).index.tolist()
    tail = pred.nsmallest(n).index.tolist()
    weight=pd.Series(0,index=factors.index)
    # weight[head]=0.5/(len(head))
    # weight[tail]=-0.5/(len(tail))
    weight[head]=((1+max_exposure)/2)/len(head)
    weight[tail]=-((1-max_exposure)/2)/len(tail)
    return weight

IF_REV = {
    'avg':True,
    'mom':True,
    'vol':False,
    'max52':True,
    'min52':True,
    'cci':True,
    'K':True,
    'D':True,
    'J':False,
    'rsi':True,
    'trix':True,
    'willr':False,
    'macd':True,
    'natr':False,
    'mfi':True
    }
def get_weight(dailyfactors,head_n=10,tail_n=10):
    
    all_weight = np.empty((0,dailyfactors.shape[0]))
    for fac in dailyfactors:
        head = dailyfactors[fac].nlargest(head_n).index.tolist()
        tail = dailyfactors[fac].nsmallest(tail_n).index.tolist()
        weight = np.zeros(351)
        weight[head] = 1 / head_n
        weight[tail] = -1 / tail_n
        if IF_REV[fac]:
            weight = weight * -1
        all_weight = np.vstack([all_weight,weight])
    sum_weight = all_weight.mean(axis=0)
    sum_weight[sum_weight>0] = sum_weight[sum_weight>0] / sum_weight[sum_weight>0].sum()
    sum_weight[sum_weight<0] = sum_weight[sum_weight<0] / sum_weight[sum_weight<0].sum() * -1
    return sum_weight