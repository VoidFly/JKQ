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

#返回为series，对应各股票
def run_strategy(df):
    '''
    stock_candidate-> dataframe
    |stockid|long_short_flag|[OPTIONAL]something like rank |
    '''
    stock_candidate=pd.DataFrame()
    ddf=df[['day','stockid','close']].set_index(['day','stockid'])['close'].unstack()
    ddf=(ddf-ddf.shift(1))/ddf.shift(1)
    stocks=ddf.rolling(3).mean().iloc[-1,:]
    max10=stocks.nlargest(10)
    min10=stocks.nsmallest(10)
    
    stock_candidate=stocks.copy()
    stock_candidate[:]=0
    stock_candidate[stocks.isin(max10)]=1
    stock_candidate[stocks.isin(min10)]=-1
    
    #check=(len(max10)!=0)| (len(min10)!=0 )
    #stock_candidate=stocks  if check else stock_candidate
    return stock_candidate

#远程返回的仓位，金额，commison是佣金
def get_position(stock_candidate,prev_pos,prev_capital,data_now,comission):
    '''
    target_pos-> np.array
    '''
    target_pos=0
    
    length=len(stock_candidate[stock_candidate!=0])
    df_now['stocks']=stock_candidate.to_list()    
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


i=0#控制seq
count=0
data_lst=[]
period=2 #eg 每两天跑一次策略
comission=0

while True:
    time.sleep(0.5)
    question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88,sequence=i))
    
    if question_response.sequence!=-1:
        #ds=np.array([x.values for x in question_response.dailystk])
        data_now=[]
        for this_stock in question_response.dailystk:
            data_now.append(this_stock.values) 
        data_lst.extend(data_now)
        
        if count%period==0:
            print('run strategy')
            #也可以考虑只取eg data_lst[-50:]
            df=pd.DataFrame(data_lst,columns=['day','stockid','open','high','low','close','volume'],dtype=int)    
            df_now=pd.DataFrame(data_now,columns=['day','stockid','open','high','low','close','volume'],dtype=int)
            #跑一次策略，返回多空备选股
            stock_candidate=run_strategy(df)
            
            #检查规则，返回持仓
            target_pos=get_position(stock_candidate,
                                    question_response.positions,
                                    question_response.capital,
                                    data_now,
                                    comission)

            # summit answer
            submit_response = contest_stub.submit_answer(contest_pb2.AnswerRequest(user_id=88,user_pin='dDTSvdwk',session_key=login_response.session_key,sequence=i,positions=target_pos))

            print(submit_response,question_response.capital)
            if not submit_response.accepted:
                print(submit_response.reason)

        i=question_response.sequence+1
        count+=1

contest_channel.close()
question_channel.close()

# #%%
# if __name__ == '__main__':
#     logging.basicConfig()
#     run()
