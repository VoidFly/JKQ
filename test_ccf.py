# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:13:42 2020

@author: 轩尘
"""
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
from strategy_utils import *

contest_channel=grpc.insecure_channel('47.103.23.116: 56702')
question_channel=grpc.insecure_channel('47.103.23.116: 56701')

contest_stub=contest_pb2_grpc.ContestStub(contest_channel)
question_stub=question_pb2_grpc.QuestionStub(question_channel)

login_response=contest_stub.login(contest_pb2.LoginRequest(user_id=88,user_pin='dDTSvdwk'))

session_key=login_response.session_key
init_capital=login_response.init_capital

i=0#控制seq
count=0
data_lst=[]
factors_lst=[]
period=1#eg 每两天跑一次策略
comission=0#
max_exposure=0.1#大盘上涨，多头增加，大盘下跌，空头增加
single_stock_position_limit=0.1#
lending_rate=0.01#
borrow_rate=0.05#
leverage=2#
factors=pd.DataFrame()
prev_factors=pd.DataFrame()#上一次计算的factors

all_return=[]


factor_select1=['avg', 'mom', 'max52', 'D', 'willr', 'natr', 'rsi']
factor_select2=['mom']
factor_select3=['mom', 'max52','D']

weights1=np.array([1/351]*351)
weights2=np.array([1/351]*351)
weights3=np.array([1/351]*351)

allselects=[factor_select1,factor_select2,factor_select3]
factor_select=allselects[0].copy()

while True:
    question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88,sequence=i))
    print(question_response.sequence)
    if question_response.sequence!=-1:
        dailystk = [x.values for x in question_response.dailystk]
        data_lst.extend(dailystk)

        if count>10:#开始不动，只要有新数据就跑一次策略
            print('run strategy')
            df=pd.DataFrame(data_lst,columns=['day','stock','open','high','low','close','volume'],
                            dtype=float).set_index(['day','stock'])
            
            dailyfactors=get_factors(df,prev_factors)  #从数据获取因子
            prev_factors=dailyfactors

            factors_lst.extend(dailyfactors.values)#向因子库追加
            #部份因子可能需要根据历史因子数据归一化
            
            
            # factor_select=select_factors(factors,n=10,period=period)  #计算相关系数选取因子
            # factor_select=['avg', 'mom', 'max52', 'cci', 'K', 'D', 'J', 'rsi', 'trix', 'willr']
            # factor_select=['avg','mom', 'vol', 'max52', 'K', 'D', 'J', 'rsi', 'trix', 'willr', 'macd', 'natr','mfi']
            # index_direction='neutral'#TODO 大盘方向，用于控制exposure
            
                
            weights = get_weight(dailyfactors[factor_select],head_n=10,tail_n=10)
            if count < 100:
                target_pos=get_position(weights,
                                        pd.DataFrame(dailystk,columns=['day','stock','open','high','low','close','volume'],dtype=float),#只需要close，待优化
                                        question_response.positions,
                                        question_response.capital / 2,
                                        comission)
            else:
                target_pos=get_position(weights,
                                        pd.DataFrame(dailystk,columns=['day','stock','open','high','low','close','volume'],dtype=float),#只需要close，待优化
                                        question_response.positions,
                                        question_response.capital,
                                        comission)
            
            #提交策略
            if count%period==0:#按周期提交
                ##summit answer
                submit_response = contest_stub.submit_answer(contest_pb2.AnswerRequest(user_id=88,user_pin='dDTSvdwk',session_key=login_response.session_key,sequence=i,positions=target_pos))

                print(submit_response,question_response.capital)
                if not submit_response.accepted:
                    print(submit_response.reason)
                    if submit_response.reason[-7:] == 'timeout':
                        i=question_response.sequence+1
                        count+=1
                        continue#如果提交超时，直接请求新数据
            
            
            #构建策略二、三的权重
            if count>=21:
                #使用前天的factor值计算权重
                factor_temp=pd.DataFrame(factors_lst[-3*351:-2*351],columns=dailyfactors.columns,dtype=float)
                weights1=get_weight(factor_temp[factor_select1],head_n=10,tail_n=10)
                weights2 = get_weight(factor_temp[factor_select2],head_n=10,tail_n=10)
                weights3 = get_weight(factor_temp[factor_select3],head_n=10,tail_n=10)
            
                #使用今天的收益率
                dayreturn=np.array(np.log(df['close'].iloc[-351:]))-np.array(np.log(df['close'].iloc[-702:-351]))
                returns=np.array([np.dot(weights1,dayreturn),np.dot(weights2,dayreturn),np.dot(weights3,dayreturn)])
                all_return.append(returns)
                
                if count%63==0:
                    select_mean=np.array(all_return[-126:]).mean(axis=0)
                    factor_select=allselects[np.argmax(select_mean)]
                    print('修改策略为过去126天中跑的最好的:"+str(factor_select)')
                
        i=question_response.sequence+1
        count+=1

        # if count==20:
            # break

    time.sleep(1)