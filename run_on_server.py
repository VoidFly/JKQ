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

df = pd.read_csv('./data/saved_data.csv',dtype=float).set_index(['day','stock'])
o = df['open'].unstack(level=1).values
h = df['high'].unstack(level=1).values
l = df['low'].unstack(level=1).values
c = df['close'].unstack(level=1).values
v = df['volume'].unstack(level=1).values
r = df['close'].unstack(level=1).pct_change().values

dailyfactors = get_factors_with_ohlcv(o,h,l,c,v)

history_weights = np.empty((0,dailyfactors.shape[1],dailyfactors.shape[0]))
for j in range(52,0,-1):
    dailyfactors = get_factors_with_ohlcv(o[-j-252:-j],h[-j-252:-j],l[-j-252:-j],c[-j-252:-j],v[-j-252:-j])
    daily_factor_weights = get_all_weights(dailyfactors,head_n=10,tail_n=10)
    history_weights = np.vstack([history_weights,daily_factor_weights.reshape((1,daily_factor_weights.shape[0],daily_factor_weights.shape[1]))])
dailyfactors = get_factors_with_ohlcv(o[-252:],h[-252:],l[-252:],c[-252:],v[-252:])
daily_factor_weights = get_all_weights(dailyfactors,head_n=10,tail_n=10)
history_weights = np.vstack([history_weights,daily_factor_weights.reshape((1,daily_factor_weights.shape[0],daily_factor_weights.shape[1]))])

init_weight = np.ones(daily_factor_weights.shape[0]) / daily_factor_weights.shape[0]
hr = np.empty((0,daily_factor_weights.shape[0]))
for i in range(50):
    factor_weights,hr = update_weights_history(history_weights[i-52],r[i-50],hr)
i=0#控制seq
count=0
data_lst=df.reset_index().values.tolist()
# factors_lst=[]
period=1#eg 每两天跑一次策略
comission=0#
max_exposure=0.1#大盘上涨，多头增加，大盘下跌，空头增加
single_stock_position_limit=0.1#
lending_rate=0.01#
borrow_rate=0.05#
leverage=2#
# factors=pd.DataFrame()
prev_factors=pd.DataFrame()#上一次计算的factors

while True:
    login_response=contest_stub.login(contest_pb2.LoginRequest(user_id=88,user_pin='dDTSvdwk'))

    session_key=login_response.session_key
    init_capital=login_response.init_capital
    i = 0
    try:
        while True:
            question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88,sequence=i))
            print(question_response.sequence)
            if question_response.sequence!=-1:
                if i == 0 :
                    i = question_response.sequence
                dailystk = [x.values for x in question_response.dailystk]

                daily = np.asarray(dailystk)
                o = np.vstack([o,daily[:,2]])
                h = np.vstack([o,daily[:,3]])
                l = np.vstack([o,daily[:,4]])
                c = np.vstack([o,daily[:,5]])
                v = np.vstack([o,daily[:,6]])
                if count>-1:#开始不动，只要有新数据就跑一次策略
                    print('run strategy')

                    
                    # dailyfactors=get_factors(df,prev_factors)  #从数据获取因子

                    dailyfactors = get_factors_with_ohlcv(o,h,l,c,v)
                    # prev_factors=dailyfactors

                    daily_factor_weights = get_all_weights(dailyfactors,head_n=10,tail_n=10) # shape: 351 * factors#

                    daily_trade_weights = get_trade_weights(daily_factor_weights,factor_weights,bias=0) # shape: 351 * 1


                    # factors_lst.extend(dailyfactors.values)#向因子库追加
                    #部份因子可能需要根据历史因子数据归一化

                    # factor_select=select_factors(factors,n=10,period=period)  #计算相关系数选取因子
                    # factor_select=['avg', 'mom', 'max52', 'cci', 'K', 'D', 'J', 'rsi', 'trix', 'willr']
                    # factor_select=['avg','mom', 'vol', 'max52', 'K', 'D', 'J', 'rsi', 'trix', 'willr', 'macd', 'natr','mfi']
                    # factor_select=['avg', 'mom', 'max52', 'D', 'willr', 'natr', 'rsi']

                    # index_direction='neutral'#TODO 大盘方向，用于控制exposure

                    # weights = get_weight(dailyfactors[factor_select],head_n=10,tail_n=10)
                    if count < 50:
                        target_pos=get_position(daily_trade_weights,
                                                pd.DataFrame(dailystk,columns=['day','stock','open','high','low','close','volume'],dtype=float),#只需要close，待优化
                                                question_response.positions,
                                                question_response.capital / 2,
                                                comission)
                    else:
                        target_pos=get_position(daily_trade_weights,
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
                    history_weights = np.vstack([history_weights,daily_factor_weights.reshape((1,daily_factor_weights.shape[0],daily_factor_weights.shape[1]))])
                    pctchg = c[-1,:] / c[-2,:] - 1
                    r = np.vstack([r,pctchg])
                    factor_weights,hr = update_weights_history(history_weights[-2],pctchg,hr)
                    data_lst.extend(dailystk)
                    df=pd.DataFrame(data_lst,columns=['day','stock','open','high','low','close','volume'], dtype=float).iloc[-351*100:]
                    df.to_csv('./data/saved_data.csv')
                i=question_response.sequence+1
                count+=1
            time.sleep(1)
            # if count==20:
                # break
    except Exception as e:
        print(e)
    time.sleep(1)