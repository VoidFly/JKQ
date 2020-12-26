# -*- coding: utf-8 -*-
#%%
%reload_ext autoreload
%autoreload 2
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
# import keras
# from keras.models import load_model

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

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

def get_factors(data,prev_factors=None):
    '''
    1*stock
    prev_factors
    '''
    if prev_factors is not None and not prev_factors.empty:
        prev_factors=prev_factors.set_index(['day','stock'])

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
    #min52=get_52weeklow(c)
    cci = get_cci(h,l,c)
    K,D,J = get_kdj(h,l,c)
    rsi = get_rsi(c)
    trix = get_trix(c)
    willr = get_willr(h,l,c)
    macd = get_macd(c)
    natr = get_natr(h,l,c)
    print(natr)
    mfi=get_mfi(c,h,l,v)
    print(mfi)

    result=pd.DataFrame([mom, vol, max52, cci, K, D, J, rsi, trix, willr, macd, natr,mfi],
                index=['mom', 'vol', 'max52', 'cci', 'K', 'D', 'J', 'rsi', 'trix', 'willr', 'macd', 'natr','mfi'],dtype=float).T

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
    
    #简单取并集/或交集
    for col in factors.columns:
        head.extend(factors[col].nlargest(n).index.tolist())
        tail.extend(factors[col].nsmallest(n).index.tolist())
    head=list(set(head))
    tail=list(set(tail))

    intersect=np.intersect1d(head,tail)
    head=np.setdiff1d(head,intersect)
    tail=np.setdiff1d(tail,intersect)
    #print('head num',len(head))
    #print('tail num',len(tail))

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
#%%
def dnn_weight(factors,model,n=10,max_exposure=0.09):
    factors=factors.set_index('stock')
    # pred = model.pred(factors.values)
    pred = pd.Series(model.predict(factors.values).flatten())
    head = pred.nlargest(10).index.tolist()
    tail = pred.nsmallest(10).index.tolist()
    weight=pd.Series(0,index=factors.index)
    # weight[head]=0.5/(len(head))
    # weight[tail]=-0.5/(len(tail))
    weight[head]=((1+max_exposure)/2)/len(head)
    weight[tail]=-((1-max_exposure)/2)/len(tail)
    return weight
#%%
def dnn_weight(factors,model,n=10,max_exposure=0.09):
    factors=factors.set_index('stock')
    # pred = model.pred(factors.values)
    pred = pd.Series(model.predict(factors.values).flatten())
    
    # target=pred.abs().nlargest(2*n)
    # head = target[target>0].index.tolist()
    # tail = target[target<=0].index.tolist()
    head = pred.nlargest(10).index.tolist()
    tail = pred.nsmallest(10).index.tolist()
    
    weight=pd.Series(0,index=factors.index)
    if np.abs(len(head)-len(tail))<= max_exposure/(1/(2*n) ):
        weight[head]=1/(2*n)
        weight[tail]=-1/(2*n)
    elif len(head)-len(tail)> max_exposure/(1/(2*n) ):
        weight[head]=(1+max_exposure)/(2*len(head))
        if len(tail)==0:
            tail=pred.nsmallest(n).index.tolist()
        weight[tail]=-(1-max_exposure)/(2*len(tail))
    elif len(tail)-len(head)> max_exposure/(1/(2*n) ):
        weight[tail]=-(1+max_exposure)/(2*len(tail))
        if len(head)==0:
            head=pred.nlargest(n).index.tolist()
        weight[head]=(1-max_exposure)/(2*len(head))
    return weight
# def feed_nn()
#远程返回的仓位，金额，commison是佣金
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

#%%
i=0#控制seq
count=0
data_lst=[]
factors_lst=[]
period=2 #eg 每两天跑一次策略
comission=0#
max_exposure=0.1#大盘上涨，多头增加，大盘下跌，空头增加
single_stock_position_limit=0.1#
lending_rate=0.01#
borrow_rate=0.05#
leverage=2#
factors=pd.DataFrame()
prev_factors=pd.DataFrame()#上一次计算的factors

#%%
model = load_model('my_dnn2.h5')

#%%
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

            # factor_select=select_factors(factors,n=10,period=period)  #计算相关系数选取因子
            # factor_select=['avg', 'mom', 'max52', 'cci', 'K', 'D', 'J', 'rsi', 'trix', 'willr']
            # factor_select=['mom', 'vol', 'max52', 'cci', 'K', 'D', 'J', 'rsi', 'trix', 'willr', 'macd', 'natr','mfi']
            # index_direction='neutral'#TODO 大盘方向，用于控制exposure
            # # weights=get_weight(dailyfactors[factor_select+['stock']],
            #                 # n=10,max_exposure=0.1,index_direction=index_direction)  #取出本期选出因子的因子值
                            
            # weights = dnn_weight(dailyfactors[factor_select+['stock']],model)
            # if count < 100:
            #     target_pos=get_position(weights,
            #                             pd.DataFrame(dailystk,columns=['day','stock','open','high','low','close','volume'],dtype=float),#只需要close，待优化
            #                             question_response.positions,
            #                             question_response.capital / 2,
            #                             comission)
            # else:
            #     target_pos=get_position(weights,
            #                             pd.DataFrame(dailystk,columns=['day','stock','open','high','low','close','volume'],dtype=float),#只需要close，待优化
            #                             question_response.positions,
            #                             question_response.capital,
            #                             comission)
            
            # #提交策略
            # if count%period==0:#按周期提交
            #     ##summit answer
            #     submit_response = contest_stub.submit_answer(contest_pb2.AnswerRequest(user_id=88,user_pin='dDTSvdwk',session_key=login_response.session_key,sequence=i,positions=target_pos))

            #     print(submit_response,question_response.capital)
            #     if not submit_response.accepted:
            #         print(submit_response.reason)
            #         if submit_response.reason[-7:] == 'timeout':
            #             i=question_response.sequence+1
            #             count+=1
            #             continue#如果提交超时，直接请求新数据

        i=question_response.sequence+1
        count+=1

        if count==30:
            break

    time.sleep(1)


#%%
contest_channel.close()
question_channel.close()

# #%%
# if __name__ == '__main__':
#     logging.basicConfig()
#     run()
