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
from functools import reduce

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


#%%
def get_factors(data):
    '''1*stock'''
    o = data['open'].unstack(level=1)
    stock=o.columns.values
    day=o.index[-1]
    o=o.values
    h = data['high'].unstack(level=1).values
    l = data['low'].unstack(level=1).values
    c = data['close'].unstack(level=1).values
    v = data['volume'].unstack(level=1).values

    mom=get_mom(c,3)
    vol=get_vol(c,3)
    
    max52=get_52weekhigh(c)
    min52=get_52weeklow(c)
    
    result=pd.DataFrame([mom, vol, max52, min52],
                index=['mom', 'vol', 'max52', 'min52'],dtype=float).T
    result['day']=day
    result['stock']=stock
    result['close']=c[-1]
    return result # stocks|day|factors....|close

#选n个因子,返回因子名
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

#根据因子优化出权重
def get_weight(factors,n=10,max_exposure=0.1,index_direction='neutral'):

    factors=factors.set_index('stock')
    head=[]
    tail=[]
    
    #简单取并集
    for col in factors.columns:
        head.extend(factors[col].nlargest(n).index.tolist())
        tail.extend(factors[col].nsmallest(n).index.tolist())
    head=list(set(head))
    tail=list(set(tail))
    #排除共有元素
    intersect=np.intersect1d(head,tail)
    head=np.setdiff1d(head,intersect)
    tail=np.setdiff1d(tail,intersect)
    #print('head num',len(head))
    #print('tail num',len(tail))

    # #取交集 不行。。。
    # for col in factors.columns:
    #     head.append(factors[col].nlargest(n).index.tolist())
    #     tail.append(factors[col].nsmallest(n).index.tolist())

    # head=reduce(np.intersect1d,head)
    # tail=reduce(np.intersect1d,tail)
    # print('head num',len(head))
    # print('tail num',len(tail))

    #根据大盘调整仓位占比
    weight=pd.Series(0,index=factors.index)
    if index_direction=='up':
        weight[head]=((1+max_exposure)/2)/len(head)
        weight[tail]=-((1-max_exposure)/2)/len(tail)
    elif index_direction=='down':
        weight[head]=((1-max_exposure)/2)/len(head)
        weight[tail]=-((1+max_exposure)/2)/len(tail)
    else:
        weight[head]=0.5/(len(head))
        weight[tail]=-0.5/(len(tail))

    #print('weight',sum(weight))
    return weight

#远程返回的仓位，金额，commison是佣金
def get_position(weights,dailystk,prev_pos,
                prev_capital,comission):
    '''
    target_pos-> list
    TODO comission prev_pos check
    
    '''
    close=dailystk.set_index(['stock'])['close']

    target_pos=prev_capital*weights/close

    # ratio=0.7
    # prev_pos=pd.Series(prev_pos,index=target_pos.index)
    # target_pos=target_pos*(1-ratio)+prev_pos*ratio

    target_pos=target_pos.astype(int)
    print(target_pos[target_pos!=0])
    return target_pos.tolist()

def get_weight2(factors,df):
    ret=df['close'].unstack(level=1).pct_change(1).shift(-1)
    pass


#%%
i=0#控制seq
count=0
data_lst=[]
period=2 #eg 每两天跑一次策略
comission=0#TODO
max_exposure=0.1#大盘上涨，多头增加，大盘下跌，空头增加
single_stock_position_limit=0.1#TODO
lending_rate=0.01#TODO
borrow_rate=0.05#TODO
leverage=2#TODO
factors=pd.DataFrame()

while True:
    question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88,sequence=i))
    print(question_response.sequence)
    if question_response.sequence!=-1:
        dailystk = [x.values for x in question_response.dailystk]
        data_lst.extend(dailystk)

        if count%period==0 and count>5:#刚开始不动
            print('run strategy')
            df=pd.DataFrame(data_lst,columns=['day','stock','open','high','low','close','volume'],
                            dtype=float).set_index(['day','stock'])
        
            dailyfactor=get_factors(df)  #从数据获取因子

            factors=factors.append(dailyfactor)  #向因子库追加
            
            #factor_select=select_factors(factors,n=10,period=period)  #计算相关系数选取因子
            factor_select=['mom', 'vol', 'max52', 'min52']
            index_direction='neutral'#TODO 大盘方向，用于控制exposure
            weights=get_weight(dailyfactor[factor_select+['stock']],
                            n=10,max_exposure=0.1,index_direction=index_direction)  #取出本期选出因子的因子值
            
            target_pos=get_position(weights,
                                    pd.DataFrame(dailystk,columns=['day','stock','open','high','low','close','volume'],dtype=float),#只需要close，待优化
                                    question_response.positions,
                                    question_response.capital,
                                    comission)

            ##summit answer
            submit_response = contest_stub.submit_answer(contest_pb2.AnswerRequest(user_id=88,user_pin='dDTSvdwk',session_key=login_response.session_key,sequence=i,positions=target_pos))

            print(submit_response,question_response.capital)
            if not submit_response.accepted:
                print(submit_response.reason)
                if submit_response.reason[-7:] == 'timeout':
                    i=question_response.sequence+1
                    count+=1
                    continue
        i=question_response.sequence+1
        count+=1
    time.sleep(1)

        

contest_channel.close()
question_channel.close()

# #%%
# if __name__ == '__main__':
#     logging.basicConfig()
#     run()

# %%
