import time, socket
import threading
import os, json
import ast, dill
import torch
from Log.Logging_Class import ModelLogger
from NanoChefModule import NanoChef
from BaseUtils.TCP_Node import BaseTCPNode
dir_name=time.strftime("%Y%m%d")
TOTAL_LOG_FOLDER = "{}/{}".format("Log", dir_name)
if os.path.isdir(TOTAL_LOG_FOLDER) == False:
    os.makedirs(TOTAL_LOG_FOLDER)
server_logger_obj = ModelLogger("SeqOptModule", "DEBUG", TOTAL_LOG_FOLDER)
base_tcp_node_obj = BaseTCPNode()
BUFF_SIZE=8192

model_obj_dict={}
total_algorithm_dict={}

def loadModel(filename):
    """
    load ML model to use already fitted model later depending on filename.
    
    Arguments
    ---------
    directory_path (str)
    filename (str)
    
    Returns
    -------
    return loaded_model, model_obj
    """
    fname = os.path.join(filename)
    with open(fname, 'rb') as f:
        model_obj = dill.load(f)
    return model_obj

def handle_client(client_socket, client_address, server_logger):
    try:
        server_logger.info("SeqOpt", f"{client_address} is connected.")

        data = b''
        
        while True:
            part = client_socket.recv(BUFF_SIZE)
            # print(part)
            if "finish" in part.decode("utf-8") or "success" in part.decode("utf-8"):
                break
            elif len(data)==0 or "finish" not in part.decode("utf-8") or "success" in part.decode("utf-8"):
                data += part
            else:
                raise ConnectionError("Wrong tcp message in module")
        
        # print(data)
        packet_info = str(data.decode()).split(sep="/")
        jobID, module_name, action_type, action_data, mode_type = packet_info
        
        if "moduleGeneration" == action_type:
            algorithm_dict=ast.literal_eval(action_data)
            SeqOpt_obj=NanoChef(algorithm_dict)
            model_obj_dict["{}:jobID={}".format(algorithm_dict["subject"],jobID)]=SeqOpt_obj
            total_algorithm_dict["{}:jobID={}".format(algorithm_dict["subject"],jobID)]=algorithm_dict
            # print("moduleGeneration start")
            sendData="success to module generation"
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
            # print("moduleGeneration finish")
        
        elif "suggestNextStep" == action_type:
            subject=str(action_data)
            real_cond_points, norm_cond_points=model_obj_dict["{}:jobID={}".format(subject,jobID)].suggestNextStep()
            recommended_recipe={
                "real_cond_points":real_cond_points, 
                "norm_cond_points":norm_cond_points
            }
            # print("suggestNextStep start")
            base_tcp_node_obj.checkSocketStatus(client_socket, recommended_recipe, module_name, action_type)
            # print("suggestNextStep finish")
        
        elif "registerPoint" == action_type:
            recommended_recipe=ast.literal_eval(action_data)

            subject=recommended_recipe["subject"]
            input_next_points=recommended_recipe["input_next_points"]
            norm_input_next_points=recommended_recipe["norm_input_next_points"]
            property_list=recommended_recipe["property_list"]
            input_result_list=recommended_recipe["input_result_list"]

            algorithm_dict=total_algorithm_dict["{}:jobID={}".format(subject,jobID)]
            SeqOpt_obj=NanoChef(algorithm_dict)
            previous_SeqOpt_obj=model_obj_dict["{}:jobID={}".format(subject,jobID)]

            SeqOpt_obj._norm_space.target=previous_SeqOpt_obj._norm_space.target
            SeqOpt_obj._norm_space.params=previous_SeqOpt_obj._norm_space.params
            SeqOpt_obj._norm_space.seqs=previous_SeqOpt_obj._norm_space.seqs
            SeqOpt_obj._norm_space.propertys=previous_SeqOpt_obj._norm_space.propertys
            SeqOpt_obj._real_space.target=previous_SeqOpt_obj._real_space.target
            SeqOpt_obj._real_space.params=previous_SeqOpt_obj._real_space.params
            SeqOpt_obj._real_space.seqs=previous_SeqOpt_obj._real_space.seqs
            SeqOpt_obj._real_space.propertys=previous_SeqOpt_obj._real_space.propertys
            SeqOpt_obj.best_loss_list=previous_SeqOpt_obj.best_loss_list
            SeqOpt_obj.best_mae_list=previous_SeqOpt_obj.best_mae_list
            SeqOpt_obj.best_y_list=previous_SeqOpt_obj.best_y_list
            SeqOpt_obj.SeqOpt_obj.nn_block.n_observation=len(SeqOpt_obj._norm_space.res())/len(SeqOpt_obj.reagent_seqs)*2
            print("[Before] len(SeqOpt_obj._norm_space.res())", len(SeqOpt_obj._norm_space.res()))
            print("[Before] len(SeqOpt_obj._real_space.res())", len(SeqOpt_obj._real_space.res()))
            # print("SeqOpt_obj.SeqOpt_obj.nn_block.n_observation", SeqOpt_obj.SeqOpt_obj.nn_block.n_observation)
            
            SeqOpt_obj.model_logger=ModelLogger(SeqOpt_obj.subject, SeqOpt_obj.logLevel, SeqOpt_obj.TOTAL_LOG_FOLDER)
            SeqOpt_obj.optimizer = torch.optim.Adam(SeqOpt_obj.SeqOpt_obj.parameters(), lr=SeqOpt_obj.lr)

            model_obj_dict["{}:jobID={}".format(subject,jobID)]=SeqOpt_obj

            # print("registerPoint start")
            model_obj_dict["{}:jobID={}".format(subject,jobID)].registerPoint(input_next_points, norm_input_next_points, property_list, input_result_list)
            # print("registerPoint finish")
            print("[After] len(SeqOpt_obj._norm_space.res())", len(SeqOpt_obj._norm_space.res()))
            print("[After] len(SeqOpt_obj._real_space.res())", len(SeqOpt_obj._real_space.res()))

            sendData=model_obj_dict["{}:jobID={}".format(subject,jobID)]._real_space.res()
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
            
        elif "res" == action_type:
            recommended_recipe=ast.literal_eval(action_data)

            subject=recommended_recipe["subject"]

            res_SeqOpt_obj=model_obj_dict["{}:jobID={}".format(subject,jobID)]

            print("len(SeqOpt_obj._norm_space.res())", len(res_SeqOpt_obj._norm_space.res()))
            print("len(SeqOpt_obj._real_space.res())", len(res_SeqOpt_obj._real_space.res()))

            sendData=res_SeqOpt_obj._real_space.res()
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
        
        elif "output_space" == action_type:
            subject, filename=action_data.split("&")
            model_obj_dict["{}:jobID={}".format(subject,jobID)].output_space(filename) # generate csv file
            sendData=model_obj_dict["{}:jobID={}".format(subject,jobID)]._norm_space.res()
            # sendData="finish generation of outputs for normalized conditions".encode('utf-8')
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
        
        elif "output_space_realCondition" == action_type:
            subject, filename=action_data.split("&")
            model_obj_dict["{}:jobID={}".format(subject,jobID)].output_space_realCondition(filename) # generate csv file
            sendData=model_obj_dict["{}:jobID={}".format(subject,jobID)]._real_space.res()
            # sendData="finish generation of outputs for real conditions".encode('utf-8')
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
        
        elif "output_space_property" == action_type:
            subject, filename=action_data.split("&")
            model_obj_dict["{}:jobID={}".format(subject,jobID)].output_space_property(filename) # generate csv file
            sendData=model_obj_dict["{}:jobID={}".format(subject,jobID)]._real_space.res()
            # sendData="finish generation of outputs for propertys".encode('utf-8')
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
        
        elif "saveModel" == action_type:
            subject, filename=action_data.split("&") 
            model_obj_dict["{}:jobID={}".format(subject,jobID)].savedModel(filename) # generate model object
            sendData="finish saving model".encode('utf-8')
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
        
        elif "loadModel" == action_type:
            subject, mode_type, dirname, pickle_name, algorithm_str=action_data.split("&")
            filename="{}/{}/{}/{}/{}/{}".format("Data",subject,"Object",mode_type,dirname,pickle_name)
            
            algorithm_dict=ast.literal_eval(algorithm_str)
            SeqOpt_obj=NanoChef(algorithm_dict)
            loaded_SeqOpt_obj=loadModel(filename)

            SeqOpt_obj._norm_space.target=loaded_SeqOpt_obj._norm_space.target
            SeqOpt_obj._norm_space.params=loaded_SeqOpt_obj._norm_space.params
            SeqOpt_obj._norm_space.seqs=loaded_SeqOpt_obj._norm_space.seqs
            SeqOpt_obj._norm_space.propertys=loaded_SeqOpt_obj._norm_space.propertys
            SeqOpt_obj._real_space.target=loaded_SeqOpt_obj._real_space.target
            SeqOpt_obj._real_space.params=loaded_SeqOpt_obj._real_space.params
            SeqOpt_obj._real_space.seqs=loaded_SeqOpt_obj._real_space.seqs
            SeqOpt_obj._real_space.propertys=loaded_SeqOpt_obj._real_space.propertys
            SeqOpt_obj.best_loss_list=loaded_SeqOpt_obj.best_loss_list
            SeqOpt_obj.best_mae_list=loaded_SeqOpt_obj.best_mae_list
            SeqOpt_obj.best_y_list=loaded_SeqOpt_obj.best_y_list
            SeqOpt_obj.SeqOpt_obj.nn_block.n_observation=len(SeqOpt_obj._norm_space.res())/len(SeqOpt_obj.reagent_seqs)*2
            print("len(SeqOpt_obj._norm_space.res())", len(SeqOpt_obj._norm_space.res()))
            print("len(SeqOpt_obj._real_space.res())", len(SeqOpt_obj._real_space.res()))
            # print("SeqOpt_obj.SeqOpt_obj.nn_block.n_observation", SeqOpt_obj.SeqOpt_obj.nn_block.n_observation)
            
            SeqOpt_obj.model_logger=ModelLogger(SeqOpt_obj.subject, SeqOpt_obj.logLevel, SeqOpt_obj.TOTAL_LOG_FOLDER)
            SeqOpt_obj.optimizer = torch.optim.Adam(SeqOpt_obj.SeqOpt_obj.parameters(), lr=SeqOpt_obj.lr)
            model_obj_dict["{}:jobID={}".format(subject,jobID)]=SeqOpt_obj
            total_algorithm_dict["{}:jobID={}".format(algorithm_dict["subject"],jobID)]=algorithm_dict
            
            SeqOpt_obj.model_logger.info("SeqOpt ({})".format("real_space"), "{}".format(SeqOpt_obj._real_space.res()))

            SeqOpt_obj._training(search_epoch=int(len(SeqOpt_obj._norm_space.res())/SeqOpt_obj.batchSize))

            sendData="finish loading model".encode('utf-8')
            base_tcp_node_obj.checkSocketStatus(client_socket, sendData, module_name, action_type)
        
        else:
            sendData="Error commands : {}".format(data)
            client_socket.sendall(sendData.encode("utf-8"))
            server_logger.info("Master", "Error commands: {}".format(data))
        
        server_logger.info("Master", "{}: {}".format(client_address, data))
    
    except ConnectionAbortedError as e:
        server_logger.info("Master", "{}: Connection was forcibly closed.".format(client_address))


def start_server():
    SERVER_HOST='127.0.0.1'  # permit from all interfaces
    SERVER_PORT=4001 # if you want, can change
    SERVER_ACCESS_NUM=100 # permit to accept the number of maximum client
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 20)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 20)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(SERVER_ACCESS_NUM) # permit to accept 

    print("[SeqOpt] Server on at {}:{}.".format(SERVER_HOST, SERVER_PORT))
    print("[SeqOpt] Waiting...")
    
    while True:
        # start Client handler thread (while loop, wait for client request)
        client_socket, client_address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address, server_logger_obj))
        client_thread.start()

start_server()