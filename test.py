#%%
import logging
import grpc

import contest_pb2
import contest_pb2_grpc
import question_pb2
import question_pb2_grpc
import time

def update_question():
    pass


def run():
    contest_channel=grpc.insecure_channel('47.103.23.116: 56702')
    question_channel=grpc.insecure_channel('47.103.23.116: 56701')

    contest_stub=contest_pb2_grpc.ContestStub(contest_channel)
    question_stub=question_pb2_grpc.QuestionStub(question_channel)
    
    login_response=contest_stub.login(contest_pb2.LoginRequest(user_id=88,user_pin='dDTSvdwk'))
    if not login_response.success:
        print('login failed:',login_response.reason)
        contest_channel.close()
        question_channel.close()
        return

    session_key=login_response.session_key
    init_capital=login_response.init_capital


    i=0
    while True:
        time.sleep(1)
        question_response=question_stub.get_question(question_pb2.QuestionRequest(user_id=88,sequence=i))
        print(question_response)
        contest_channel.close()
        question_channel.close()

    return question_response

logging.basicConfig()
data=run()


# #%%
# if __name__ == '__main__':
#     logging.basicConfig()
#     run()