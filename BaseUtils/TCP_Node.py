import socket
import os, sys
import json
import time

class BaseTCPNode(object):

    def __init__(self):
        self.BUFF_SIZE = 4096
    
    def checkSocketStatus(self, client_socket, res_msg, hardware_name, action_type):
        if bool(res_msg) == True:
            if type(res_msg)==dict: # 
                ourbyte=b''
                ourbyte = json.dumps(res_msg).encode("utf-8")
                self.sendTotalJSON(client_socket, ourbyte)
                # send finish message to main computer
                time.sleep(1)
                finish_msg="finish"
                client_socket.sendall(finish_msg.encode())
            if type(res_msg)==list: # 
                ourbyte=b''
                ourbyte = str(res_msg).encode("utf-8")
                self.sendTotalJSON(client_socket, ourbyte)
                # send finish message to main computer
                time.sleep(1)
                finish_msg="finish"
                client_socket.sendall(finish_msg.encode())
            else:
                cmd_string_end = "[{}] {} action success".format(hardware_name, action_type)
                client_socket.sendall(cmd_string_end.encode())
        elif bool(res_msg) == False:
            cmd_string_end = "[{}] {} action error".format(hardware_name, action_type)
            client_socket.sendall(cmd_string_end.encode())
            raise ConnectionError("{} : Please check".format(cmd_string_end))

    def callServer(self, host, port, command_byte):
        res_msg=""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # receive filename (XXXXXX.json)
            s.connect((host, port))
            s.sendall(command_byte)
            msg = b''
            while True:
                part = s.recv(self.BUFF_SIZE)
                msg += part
                if len(part) < self.BUFF_SIZE:
                    s.close()
                    break
            res_msg=msg.decode('UTF-8')
        return res_msg
    
    def sendTotalJSON(self, client_socket, ourbyte):
        cnt=0
        while (cnt+1)*self.BUFF_SIZE < len(ourbyte):
            msg_temp = b""+ourbyte[cnt * self.BUFF_SIZE: (cnt + 1) * self.BUFF_SIZE]
            # print("length : {}".format(len(msg_temp)))
            # print(msg_temp)
            client_socket.sendall(msg_temp)
            cnt += 1
        msg_temp = b"" + ourbyte[cnt * self.BUFF_SIZE: len(ourbyte)]
        client_socket.sendall(msg_temp)