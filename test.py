import logging
import grpc

import contest_pb2
import contest_pb2_grpc
import question_pb2
import question_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    # with grpc.insecure_channel('localhost:50051') as channel:
    #     stub = helloworld_pb2_grpc.GreeterStub(channel)
    #     response = stub.SayHello(helloworld_pb2.HelloRequest(name='you'))
    #     print("Greeter client received: " + response.message)
    #     response=stub.SayHelloAgain(helloworld_pb2.HelloRequest(name='you'))
    #     print("Greeter client received: " + response.message)

    with grpc.insecure_channel('47.103.23.116: 56702') as contest_channel: 
        stub=contest_pb2_grpc.ContestStub(contest_channel)
        login_response=stub.login(contest_pb2.LoginRequest(user_id=88,user_pin='dDTSvdwk'))
        print(login_response)
        
    #question_channel=grpc.insecure_channel('47.103.23.116: 56701')


if __name__ == '__main__':
    logging.basicConfig()
    run()