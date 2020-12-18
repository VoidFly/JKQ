# -*- coding: utf-8 -*-
#%%

import logging
import grpc
import pandas as pd
import sys
sys.path.append('proto')
from proto import contest_pb2
from proto import contest_pb2_grpc
from proto import question_pb2
from proto import question_pb2_grpc
import numpy as np
import time
from utils import *


contest_channel=grpc.insecure_channel('47.103.23.116: 56702')
question_channel=grpc.insecure_channel('47.103.23.116: 56701')

contest_stub=contest_pb2_grpc.ContestStub(contest_channel)
question_stub=question_pb2_grpc.QuestionStub(question_channel)

login_response=contest_stub.login(contest_pb2.LoginRequest(user_id=88,user_pin='dDTSvdwk'))
# if not login_response.success:
#     print('login failed:',login_response.reason)
#     contest_channel.close()
#     question_channel.close()
#     return

session_key=login_response.session_key
init_capital=login_response.init_capital

# #返回为series，对应各股票
# def run_strategy(df):
#     '''
#     stock_candidate-> dataframe
#     |stockid|long_short_flag|[OPTIONAL]something like rank |
#     '''
#     stock_candidate=pd.DataFrame()
#     ddf=df[['day','stockid','close']].set_index(['day','stockid'])['close'].unstack()
#     ddf=(ddf-ddf.shift(1))/ddf.shift(1)
#     stocks=ddf.rolling(3).mean().iloc[-1,:]
#     max10=stocks.nlargest(10)
#     min10=stocks.nsmallest(10)
    
#     stock_candidate=stocks.copy()
#     stock_candidate[:]=0
#     stock_candidate[stocks.isin(max10)]=1
#     stock_candidate[stocks.isin(min10)]=-1
    
#     #check=(len(max10)!=0)| (len(min10)!=0 )
#     #stock_candidate=stocks  if check else stock_candidate
#     return stock_candidate



#%%
def get_factors(data):
    '''1*stock'''
    o = data['open'].unstack()
    stocks=o.columns.values
    day=o.index[-1]
    o=o.values
    h = data['high'].unstack().values
    l = data['low'].unstack().values
    c = data['close'].unstack().values
    v = data['volume'].unstack().values

    mom=get_mom(c,3)
    vol=get_vol(c,3)
    
    max52=get_52weekhigh(c)
    min52=get_52weeklow(c)
    
    result=pd.DataFrame([mom, vol, max52, min52],
                index=['mom', 'vol', 'max52', 'min52'],dtype=float).T
    result['day']=day
    result['stocks']=stocks
    result['close']=c[-1]
    return result # stocks|day|factors|close

#选n个因子,返回因子名
def select_factors(factors,n=10):
    factors=factors.set_index(['day','stocks'])
    factor_names=factors.columns.drop(['close'])
    
    c=factors['close'].unstack().shift(1)
    def spearman(x):
        spear=x.unstack().rank(axis=1)-c.rank(axis=1)
        spearcorrs=6*(spear*spear).sum(axis=1)/351/(351*351-1)
        return spearcorrs.mean()
    ic=factors[factor_names].apply(lambda x:spearman(x))#lambda x:x.unstack(),axis=1)
    factor_select=ic.nlargest(n).index.to_list()
    
    return factor_select

#根据因子优化出权重
def get_weight(factors,n=10,index_direction='neutral'):

    factors=factors.set_index('stock')
    head=[]
    tail=[]
    
    #简单取并集/或交集
    for col in factors.columns:
        head.extend(factors[col].nlargest(n).index.tolist())
        tail.extend(factors[col].nsmallest(n).index.tolist())
    head=list(set(head))
    tail=list(set(tail))

    #理想仓位占比
    weight=pd.Series(0,index=factors.index)
    if index_direction=='up':
        weight[head]=(1+max_exposure)/(1-max_exposure)
        weight[tail]=-1
    elif index_direction=='down':
        weight[head]=1
        weight[tail]=-(1-max_exposure)/(1+max_exposure)
    else:
        weight[head]=1
        weight[tail]=-1

    return weight

#远程返回的仓位，金额，commison是佣金
def get_position(weights,prev_pos,prev_capital,data_now,comission):
    '''
    target_pos-> np.array
    '''
    target_pos=0
    
    length=len(weights[weights!=0])

    df_now['stocks']=weights.to_list()    
    df_now['stocks']=df_now['stocks']/df_now['close']

    print(sum(df_now['stocks']),prev_capital,length)
    
    temp=pd.Series(df_now.set_index('stockid')['stocks'])
    #df_now['stocks1']=df_now['stocks']*prev_capital/length
    #df_now['stocks-1']=-df_now['stocks']*prev_capital/length
    if length!=0:
        temp=temp*prev_capital/length
    temp=temp.apply(lambda x:int(x))
    
    target_pos=temp.to_list()
    m=pd.Series(target_pos)
    print(m[m!=0])
    return target_pos

#%%

i=0#控制seq
count=0
data_lst=[]
period=2 #eg 每两天跑一次策略
comission=0
max_exposure=0.1#大盘上涨，多头增加，大盘下跌，空头增加
single_stock_position_limit=0.1
lending_rate=0.01
borrow_rate=0.05
leverage=2
factors=pd.DataFrame()

while True:
    time.sleep(0.5)
    question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88,sequence=i))
    if question_response.sequence!=-1:
        dailystk = [x.values for x in question_response.dailystk]
        data_lst.extend(dailystk)

        if count%period==0:
            print('run strategy')
            #也可以考虑只取eg data_lst[-50:]
            df=pd.DataFrame(data_lst,columns=['day','stockid','open','high','low','close','volume'],dtype=int)    
            
            dailyfactor=get_factors(df)  #从数据获取因子

            facotrs=factors.append(dailyfactor)  #向因子库追加
            
            factor_select=select_factors(factors)  #计算相关系数选取因子

            index_direction='neutral'#大盘方向，需要不断更新
            weights=get_weight(dailyfactor[factor_select+['stock']])  #取出本期选出因子的因子值
            
            target_pos=get_position(weights,
                                    pd.DataFrame(dailystk,columns=['day','stockid','open','high','low','close','volume'],dtype=float),
                                    question_response.positions,
                                    question_response.capital,
                                    max_exposure,
                                    index_direction,
                                    comission)

            # summit answer
            # submit_response = contest_stub.submit_answer(contest_pb2.AnswerRequest(user_id=88,user_pin='dDTSvdwk',session_key=login_response.session_key,sequence=i,positions=target_pos))

            # print(submit_response,question_response.capital)
            # if not submit_response.accepted:
            #     print(submit_response.reason)

        i=question_response.sequence+1
        count+=1

contest_channel.close()
question_channel.close()

# #%%
# if __name__ == '__main__':
#     logging.basicConfig()
#     run()
