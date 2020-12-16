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
import time
import numpy as np


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


def run_strategy(df):
    '''
    stock_candidate-> dataframe
    |stockid|long_short_flag|[OPTIONAL]something like rank |
    shape(num_of_stocks,3)
    '''
    stock_candidate=pd.DataFrame()
    return stock_candidate

def get_position(stock_candidate,prev_pos,prev_capital,comission):
    '''
    target_pos-> np.array
    '''
    target_pos=0
    
    return target_pos


i=0#控制seq
count=0
data_lst=[]
period=2 #eg 每两天跑一次策略
comission=0

while count<5:
    time.sleep(0.5)
    question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88,sequence=i))
    print(question_response.sequence)

    if question_response.sequence!=-1:
        #ds=np.array([x.values for x in question_response.dailystk])
        for this_stock in question_response.dailystk:
            data_lst.append(this_stock.values)

        if count%2==0:
            print('run strategy')
            #也可以考虑只取eg data_lst[-50:]
            df=pd.DataFrame(data_lst,columns=['day','stockid','open','high','low','close','volume'],dtype=int)
            
            #跑一次策略，返回多空备选股
            stock_candidate=run_strategy(df)#candi for candicate

            #检查规则，返回持仓
            target_pos=get_position(stock_candidate,
                                    question_response.positions,
                                    question_response.capital,
                                    comission)

            # summit answer
            submit_response = contest_stub.submit_answer(contest_pb2.AnswerRequest(user_id=88,user_pin='dDTSvdwk',session_key=login_response.session_key,sequence=i,positions=target_pos))
            print(submit_response)
            

        i=question_response.sequence+1
        count+=1

contest_channel.close()
question_channel.close()

# #%%
# if __name__ == '__main__':
#     logging.basicConfig()
#     run()
