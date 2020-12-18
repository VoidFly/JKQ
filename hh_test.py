# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:14:05 2020

@author: 轩尘
"""
import logging
import grpc
import pandas as pd
import numpy as np
import time
import sys
from proto import common_pb2,common_pb2_grpc
from proto import contest_pb2,contest_pb2_grpc,question_pb2,question_pb2_grpc
from utils import *

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
def select_factors(factors,n):
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
def get_weight(factors):
    pass
#%%

contest_channel=grpc.insecure_channel('47.103.23.116: 56702')
question_channel=grpc.insecure_channel('47.103.23.116: 56701')

contest_stub=contest_pb2_grpc.ContestStub(contest_channel)
question_stub=question_pb2_grpc.QuestionStub(question_channel)

login_response=contest_stub.login(contest_pb2.LoginRequest(user_id=88,user_pin='dDTSvdwk'))

sequence = 0
last_sequence = 0
data_lst=[]
factors=pd.DataFrame()
while sequence < 1000:
    question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88))
    sequence = question_response.sequence
    if last_sequence < sequence:
        # Run our strategy here, resulting in a POS array with length 351

        dailystk = [x.values for x in question_response.dailystk]
        #ds = np.asarray(dailystk)
        data_lst.extend(dailystk)
        df=pd.DataFrame(data_lst,columns=['day','stockid','open','high','low','close','volume'],dtype=float)
        
        dailyfactor=get_factors(df)  #从数据获取因子
        facotrs=factors.append(dailyfactor)  #向因子库追加
        
        factor_select=select_factors(factors,n)  #计算相关系数选取因子
        weight=get_weight(dailyfactor[factor_select+['stock']])  #取出本期选出因子的因子值

        #factors.append()
        # Here is our target array
        # pos = np.sign(np.random.rand(351) - 0.5) * np.floor(question_response.capital / 351 / ds[:,5])
        # pos = pos * -1
        # summit answer
        # submit_response = contest_stub.submit_answer(contest_pb2.AnswerRequest(user_id=88,user_pin='dDTSvdwk',session_key=login_response.session_key,sequence=sequence,positions=pos))
        #print(submit_response)
        time.sleep(1)
        break
    
    
    
    
    