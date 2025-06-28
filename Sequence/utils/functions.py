import numpy as np
import pandas as pd
import copy, json
import torch


def loss_fn(y_pred, y_real):
    return torch.mean((y_pred-y_real)**2)

def caculateMAE(input_surface_name:str, input_search_epoch:int, input_n_search_epochs:int, 
                input_BNN_obj:object, input_model_logger:object, input_target_space:list, input_device:str):
    surface_name=input_surface_name
    search_epoch=input_search_epoch
    n_search_epochs=input_n_search_epochs
    model_logger=input_model_logger
    target_space=input_target_space
    device=input_device
    target_info_values=torch.Tensor([]).to(device)

    input_x_tensors=torch.Tensor([]).to(device)
    target_info_values=torch.Tensor([]).to(device)
    """
    extract x, y tensor in target space
    """
    for target_info in target_space:
        target_x_value=torch.Tensor(target_info["x"]).to(device)
        input_x_tensors=torch.cat((input_x_tensors, torch.Tensor(target_x_value).unsqueeze(0)), dim=0)
        target_info_value=torch.Tensor(target_info["target"]).to(device)
        target_info_values=torch.cat((target_info_values, torch.Tensor(target_info_value).unsqueeze(0)), dim=0)

    mu_variables, sigma_variables = input_BNN_obj(input_x_tensors)

    mu_variables=mu_variables.cpu().data.numpy()
    target_info_values=target_info_values.cpu().data.numpy()
    y_real_pred = np.vstack((target_info_values.astype(float).flatten(), mu_variables.flatten())).transpose()
    model_logger.info("model ({}-{}/{})-training".format(surface_name,search_epoch,n_search_epochs), "total prediction result --> mu: {}".format(mu_variables.flatten()))
    model_logger.info("model ({}-{}/{})-training".format(surface_name,search_epoch,n_search_epochs), "total prediction result --> sigma: {}".format(sigma_variables.flatten()))
    model_logger.info("model ({}-{}/{})-training".format(surface_name,search_epoch,n_search_epochs), "y_real/y_pred: {}".format(y_real_pred))
    model_logger.info("model ({}-{}/{})-training".format(surface_name,search_epoch,n_search_epochs), "Final MAE: {}".format(np.sum(np.abs(mu_variables.flatten()-target_info_values.flatten()))/len(mu_variables.flatten())))

    return y_real_pred, np.sum(np.abs(mu_variables.flatten()-target_info_values.flatten()))/len(target_info_values.flatten())

def train_seq(input_surface_name:str, input_search_epoch:int, input_n_search_epochs: int, 
        input_seq_opt_obj:object, input_optimizer:object, input_model_logger:object, input_target_space:list, 
        input_patience:float, input_n_train_epochs:int, input_device:str):
    """
    # Early stopping 관련 변수 설정
    """
    surface_name=input_surface_name
    search_epoch=input_search_epoch
    n_search_epochs=input_n_search_epochs
    trained_seq_opt_obj=input_seq_opt_obj 
    optimizer=input_optimizer
    model_logger=input_model_logger
    best_loss = float('inf') 
    target_space=input_target_space
    patience=input_patience
    n_train_epochs=input_n_train_epochs
    device=input_device

    best_loss_epoch=0
    best_model_state_dict = None
    best_model_obj=None
    current_patience = 0

    seq_tensors=[]
    x_tensors=[]
    target_info_values=torch.Tensor([]).to(device)
    for target_info in target_space:
        seq_tensors.append(target_info["seq"])
        x_tensors.append(target_info["x"])
        target_info_value=torch.Tensor(target_info["target"]).to(device)
        target_info_values=torch.cat((target_info_values, torch.Tensor(target_info_value).unsqueeze(0)), dim=0)
    for epoch in range(n_train_epochs):
        mu_variables=torch.Tensor([]).to(device)
        sigma_variables=torch.Tensor([]).to(device)
        # print("seq_tensors,x_tensors", seq_tensors,x_tensors)
        mu_variables, sigma_variables = trained_seq_opt_obj(seq_tensors,x_tensors) # make a prediction
        loss = loss_fn(mu_variables, target_info_values)
        optimizer.zero_grad() # initialize gradient
        loss.backward() # execute backward
        optimizer.step() # optimizer update parameter depending on definec update rule
        if loss < best_loss: # best_loss보다 loss가 더 작은 값이 나옴?
            best_loss = loss.item()
            best_loss_epoch=epoch
            current_patience = 0
            best_model_state_dict=trained_seq_opt_obj.state_dict()
            best_model_obj=copy.deepcopy(trained_seq_opt_obj)
        else: # best_loss가 여전히 loss 작다면?
            current_patience += 1
            if current_patience >= patience:
                model_logger.info("model ({}-{}/{})-{}".format(surface_name,search_epoch,n_search_epochs,device), f'Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.4f}')
                break
        if epoch % 1000 == 0:
            model_logger.info("model ({}-{}/{})-{}".format(surface_name,search_epoch,n_search_epochs,device), "epoch: {} / loss: {}".format(epoch, best_loss))
    model_logger.info("model ({}-{}/{})-{}".format(surface_name,search_epoch,n_search_epochs,device), "Final epoch: {} / loss: {}".format(best_loss_epoch, best_loss))

    return best_model_obj, best_loss, best_model_state_dict

# 2. Loss Function
def evidential_loss(y, mu, alpha, beta, nu, lambda_reg=1e-3):
    # NLL for evidential regression
    two_bl = 2 * beta * (1 + nu)
    log_likelihood = (
        0.5 * torch.log(np.pi / nu)
        - alpha * torch.log(two_bl)
        + (alpha + 0.5) * torch.log(two_bl + nu * (y - mu) ** 2)
    )
    nll = torch.mean(log_likelihood)

    # Regularization term
    reg = torch.mean((y - mu) ** 2 * (2 * nu + alpha))
    return nll + lambda_reg * reg

def caculateMAE_seq(input_surface_name:str, input_search_epoch:int, input_n_search_epochs:int, 
                input_seq_opt_obj:object, input_model_logger:object, input_target_space:list, input_device:str):
    
    surface_name=input_surface_name
    search_epoch=input_search_epoch
    n_search_epochs=input_n_search_epochs
    model_logger=input_model_logger
    target_space=input_target_space
    device=input_device

    mu_variables=torch.Tensor([]).to(device)
    sigma_variables=torch.Tensor([]).to(device)
    target_info_values=torch.Tensor([]).to(device)
    """
    extract x, y tensor in target space
    """
    seq_tensors=[]
    x_tensors=[]
    target_info_values=torch.Tensor([]).to(device)
    for target_info in target_space:
        seq_tensors.append(target_info["seq"])
        x_tensors.append(target_info["x"])
        target_info_value=torch.Tensor(target_info["target"]).to(device)
        target_info_values=torch.cat((target_info_values, torch.Tensor(target_info_value).unsqueeze(0)), dim=0)

    mu_variables, sigma_variables = input_seq_opt_obj(seq_tensors,x_tensors) # make a prediction

    print("target_info_values.shape:", target_info_values.shape)
    print("mu_variables.shape:", mu_variables.shape)

    y_real_pred = torch.cat((target_info_values.float().view(-1, 1), mu_variables.view(-1, 1)), dim=1)
    mae=torch.sum(torch.abs(mu_variables.view(-1) - target_info_values.view(-1))) / len(mu_variables.view(-1))
    model_logger.info("SeqOpt ({}-{}/{})-training".format(surface_name, search_epoch, n_search_epochs), "total prediction result --> mu: {}".format(mu_variables.view(-1)))
    model_logger.info("SeqOpt ({}-{}/{})-training".format(surface_name, search_epoch, n_search_epochs), "total prediction result --> sigma: {}".format(sigma_variables.view(-1)))
    model_logger.info("SeqOpt ({}-{}/{})-training".format(surface_name, search_epoch, n_search_epochs), "y_real/y_pred: {}".format(y_real_pred))
    model_logger.info("SeqOpt ({}-{}/{})-training".format(surface_name, search_epoch, n_search_epochs), "Final MAE: {}".format(mae))

    return y_real_pred.cpu().data.numpy(), mae.cpu().data.numpy()

def caculateMAE_seq_real(input_experiment_name:str, input_search_epoch:int,
                input_seq_opt_obj:object, input_model_logger:object, 
                input_target_space:list, input_device:str):
    experiment_name=input_experiment_name
    search_epoch=input_search_epoch
    model_logger=input_model_logger
    target_space=input_target_space
    device=input_device

    mu_variables=torch.Tensor([]).to(device)
    sigma_variables=torch.Tensor([]).to(device)
    target_info_values=torch.Tensor([]).to(device)
    """
    extract x, y tensor in target space
    """
    seq_tensors=target_space.seqs
    x_tensors=target_space.params
    target_array=target_space.target.reshape(-1, 1)
    target_info_values=torch.Tensor(target_array).to(device)

    mu_variables, sigma_variables = input_seq_opt_obj(seq_tensors,x_tensors) # make a prediction

    y_real_pred = torch.cat((target_info_values.float().view(-1, 1), mu_variables.view(-1, 1)), dim=1)
    mae=torch.sum(torch.abs(mu_variables.view(-1) - target_info_values.view(-1))) / len(mu_variables.view(-1))
    model_logger.info("SeqOpt ({}) iter:{}-device:{}-pred".format(experiment_name, search_epoch, device), "total prediction result --> mu: {}".format(mu_variables.view(-1)))
    model_logger.info("SeqOpt ({}) iter:{}-device:{}-pred".format(experiment_name, search_epoch, device), "total prediction result --> sigma: {}".format(sigma_variables.view(-1)))
    model_logger.info("SeqOpt ({}) iter:{}-device:{}-pred".format(experiment_name, search_epoch, device), "y_real/y_pred: {}".format(y_real_pred))
    model_logger.info("SeqOpt ({}) iter:{}-device:{}-pred".format(experiment_name, search_epoch, device), "Final MAE: {}".format(mae))

    return y_real_pred.cpu().data.numpy(), mae.cpu().data.numpy()

def load_json_to_dict(file_path):
    with open(file_path, "r") as json_file:
        json_string = json_file.read()
        json_dict = json.loads(json_string)
    return json_dict
    
def output_space_property(target_space, dirname, filename):
    """
    Parameters
    ----------
    dirname (str) :"DB/2022XXXX
    filename : "{}_data" + .csv
    
    Returns
    -------
    None
    """
    target_space_data=[]
    target_space_columns=[]
    for target_data in target_space:
        target_space_columns=list(target_data.keys())
        target_space_data.append(list(target_data.values()))
    total_path="{}/{}.csv".format(dirname, filename)
    df = pd.DataFrame(data=target_space_data, columns=target_space_columns)
    df.to_csv(total_path, index=False)
    