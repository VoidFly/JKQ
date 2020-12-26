# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:46:56 2020

@author: 轩尘
"""
#%%
import numpy as np
import pandas as pd

riskfreeRate=0.01
bound=0.03
period=1
turnovercost=0.0005
    

def portbaseinfor(weight):
    weight = weight.unstack()
    turnover = weight.fillna(0).diff().abs().sum(axis=1)/2
    inforbydate = pd.concat([weight.count(axis=1),weight.idxmax(axis=1),weight.max(axis=1),weight.idxmin(axis=1),weight.min(axis=1),turnover],\
                            axis=1,keys=['count','maxstock','maxweight','minstock','minweight','turnover'],sort=True)
    inforbyasset = pd.concat([weight.count(),weight.mean(),weight.max(),weight.min()],axis=1,keys=['holdingdays','mean','max','min'],sort=True)
    inforbyasset.sort_values(by='holdingdays',ascending=False,inplace=True)
    return {'bydate':inforbydate,'byasset':inforbyasset}
    
def portevaluation(df):
    result = pd.Series(dtype=float)
    lenth = len(df)    
    result.loc["Annualized Returns"] = (np.cumprod(df["AbRet"]+1).iloc[-1])**(252/lenth)-1
    result.loc['PtfNetValueRatio']=df['PtfNetValue'].iloc[-1]/df['PtfNetValue'].iloc[0]
    result.loc['BenchNetValueRatio']=df['BenchNetValue'].iloc[-1]/df['BenchNetValue'].iloc[0]
    result.loc['NetValueDiff']=result.loc['PtfNetValueRatio']-result.loc['BenchNetValueRatio']
    result.loc["Max Drawdown"] = df["MaxDrawDown"].max()
    result.loc["Sharpe Ratio"] = df["AbRet"].mean()/df["AbRet"].std()*np.sqrt(252)
    result.loc["Volatility"] = df["AbRet"].std()*np.sqrt(252)
    result.loc["Turnover"] = df['turnover'].mean()    
    result.loc["WinRate"] = (df["AbRet"]>0).sum()/df["AbRet"].count()
    result.loc['Calmar Ratio']=result.loc["Annualized Returns"]/result.loc["Max Drawdown"]
    dfup=df[df["AbRet"]>0]["AbRet"]
    dfdown=df[df["AbRet"]<0]["AbRet"]
    result.loc['ProfitLoss Ratio']=-dfup.sum()/dfdown.sum()
    result.loc['Downside Risk']=dfdown.std()*np.sqrt(252)
    result.loc['Sortino Ratio(risk-free rate=%.3f)'%(riskfreeRate)]=(result.loc["Annualized Returns"]-riskfreeRate)/result.loc['Downside Risk']
    result.loc["HighAbRetRate(AbRet>%.3f)"%(bound)] = (df['AbRet']>bound).sum()/df["AbRet"].count()#高超额收益的days所占比率
    result.loc["LowAbRetRate(AbRet<-%.3f)"%(bound)] = (df['AbRet']<-bound).sum()/df["AbRet"].count()
    result.loc['Time periods']=str(df.index[0])+' to '+str(df.index[-1])
    result.loc['Time length']=len(df)
    return result

def backtest(weights,price,bench_code='000905.SH'):
    weights=weights.set_index(['date','asset'])
    baseinfor =portbaseinfor(weights)
    adjclose = price.pivot(index='date',columns='asset',values='adjclose')
    pctchange = adjclose.pct_change(period).dropna(axis=0,how="all")*100
    pctchange = pctchange.unstack().reset_index()
    pctchange.columns = ['asset','date','ret']

    weights=weights.reset_index().pivot(index = 'date',\
                               columns = 'asset',\
                               values = 'weight').fillna(0).shift(2) 
    weights=weights.dropna(how='all').unstack().reset_index()

    #计算每天的表现
    df = pd.merge(weights,pctchange,on=['date','asset'])

    df.columns = ['asset','date','weight', 'ret']

    df['ret'] = df['weight']*df['ret']

    ret = df.groupby('date').sum()['ret'].rename('PtfRet')

    bench_ret = pctchange.groupby('date').mean()['ret'].rename('BenchRet')
    Ret = pd.concat([ret,bench_ret],axis=1,sort = False).dropna()#组合回报率的index为得到收益的日期即pctchange对应的日期
    
    Ret["turnover"] = baseinfor['bydate']['turnover'].shift(2)
    Ret['PtfRet'] = Ret['PtfRet']/100-Ret['turnover']*turnovercost #减去千1.5的换手成本
    Ret['BenchRet'] = Ret['BenchRet']/100    
    Ret["AbRet"] = Ret["PtfRet"]-Ret["BenchRet"]
    Ret["NetValue"] = np.cumprod(Ret["AbRet"]+1)
    Ret["PtfNetValue"] = np.cumprod(Ret["PtfRet"]+1)
    Ret["BenchNetValue"] = np.cumprod(Ret["BenchRet"]+1)
    Ret["MaxDrawDown"] = 1-Ret["NetValue"]/(Ret["NetValue"].cummax())
        
    #计算策略评价指标
    result =portevaluation(Ret)
    Result = {}
    Result['Performance'] = Ret
        
    Result['Assess'] = result
    upresult = portevaluation(Ret[Ret['BenchRet']>0])
    downresult = portevaluation(Ret[Ret['BenchRet']<=0])
    Result['BullAssess'] = upresult
    Result['BearAssess'] = downresult
    Result['baseinfor'] = baseinfor
    
    return Result

#%%
price=pd.read_csv('./data/sample_data.csv',encoding='gbk')
price = price[['天数','股票代码','收盘价']]
price.columns=['date','asset','adjclose']
weight=price[['date','asset']].copy()
weight['weight']=1/500
result=backtest(weight,price,bench_code='000905.SH')
# %%
