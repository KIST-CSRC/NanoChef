from scipy.stats import qmc
import numpy as np
import matplotlib
matplotlib.use('Agg')

def latin_hypercube_sampling(n, m, seed=None):
    """
    n차원에서 m개의 샘플을 생성하는 라틴 하이퍼큐브 샘플링 함수.

    Parameters:
    n (int): 차원의 수.
    m (int): 샘플의 수.
    seed (int, optional): 난수 시드 값 (재현성을 위해 사용). 기본값은 None.

    Returns:
    numpy.ndarray: (m, n) 크기의 라틴 하이퍼큐브 샘플 배열.
    """
    sampler = qmc.LatinHypercube(d=n, seed=seed)  # LHS 샘플러 초기화
    sample = sampler.random(m)  # m개의 샘플 생성
    
    return sample

def acq_max_virtual(seq_opt_obj, model_logger, input_seq_points, input_X_points, batch_size, kappa=0.1, device="cpu"):
    mu_variable, sigma_variable=seq_opt_obj(input_seq_points, input_X_points)

    selected_mu_variable = mu_variable.cpu().data.numpy()
    selected_sigma_variable = sigma_variable.cpu().data.numpy()

    lcb_result=np.array(selected_mu_variable)-np.array(selected_sigma_variable)*kappa
    lcb_result=lcb_result.flatten()

    sort_lcb_index=np.argsort(lcb_result)[::-1][-batch_size:]
    non_lcb_index=np.argsort(lcb_result)[::-1][:batch_size]
    # model_logger.info("model-acq", "total prediction result --> pi: {}, mu: {}, sigma: {}".format(total_pi_variable, total_mu_variable, total_sigma_variable))
    model_logger.info("model-acq", "selected lower_confidence_bound index, result: {}, {}".format(sort_lcb_index, np.array(lcb_result)[sort_lcb_index]))
    model_logger.info("model-acq", "selected lower_confidence_bound result --> mu: {}, sigma: {}".format(np.array(selected_mu_variable)[sort_lcb_index], np.array(selected_sigma_variable)[sort_lcb_index]))
    model_logger.info("model-acq", "selected next data points, seq: {}, cond: {}".format(np.array(input_seq_points)[sort_lcb_index], np.array(input_X_points)[sort_lcb_index]))

    model_logger.info("model-acq", "non-selected lower_confidence_bound index, result: {}, {}".format(non_lcb_index,np.array(lcb_result)[non_lcb_index]))
    model_logger.info("model-acq", "non-selected lower_confidence_bound result --> mu: {}, sigma: {}".format(np.array(selected_mu_variable)[non_lcb_index], np.array(selected_sigma_variable)[non_lcb_index]))
    model_logger.info("model-acq", "non-selected next data points, seq: {}, cond: {}".format(np.array(input_seq_points)[non_lcb_index], np.array(input_X_points)[non_lcb_index]))

    return np.array(input_seq_points)[sort_lcb_index], np.array(input_X_points)[sort_lcb_index]


def acq_max_real(acq_method:str, seq_opt_obj, model_logger, space, reagent_seqs, acq_n_samples, batch_size, kappa=0.05, device="cpu"):
    # start n_acq_sampling  
    samplingrng = space.bounds.tolist()
    sampling_cond_list=latin_hypercube_sampling(len(samplingrng), acq_n_samples)
    sampling_cond_array=[]
    for sampling_cond in sampling_cond_list:
        round_sampling_cond=space._bin(sampling_cond)
        sampling_cond_array.append(round_sampling_cond)
    sampling_cond_array=np.array(sampling_cond_array)
    
    # Create a new array by repeating the array for len(reagent_seqs) times
    total_x = np.tile(sampling_cond_array, (len(reagent_seqs), 1))
    total_seq = []
    for reagent_seq in reagent_seqs:
        for i in range(len(sampling_cond_array)):
            total_seq.append(reagent_seq)
            
    # prediction
    mu_variable, sigma_variable=seq_opt_obj(total_seq, total_x)
    selected_mu_variable = mu_variable.cpu().data.numpy()
    selected_sigma_variable = sigma_variable.cpu().data.numpy()
    
    # acquisition function
    if acq_method=="lcb":
        result=np.array(selected_mu_variable)-np.array(selected_sigma_variable)*kappa
        result=result.flatten()
    sort_index=np.argsort(result)[::-1][-batch_size:]
    non_index=np.argsort(result)[::-1][:batch_size]

    # model_logger.info("model-acq", "total prediction result --> pi: {}, mu: {}, sigma: {}".format(total_pi_variable, total_mu_variable, total_sigma_variable))
    model_logger.info("model-acq ({})".format(device), "selected lower_confidence_bound index, result: {}, {}".format(sort_index, np.array(result)[sort_index]))
    model_logger.info("model-acq ({})".format(device), "selected lower_confidence_bound result --> mu: {}, sigma: {}".format(np.array(selected_mu_variable)[sort_index], np.array(selected_sigma_variable)[sort_index]))
    model_logger.info("model-acq ({})".format(device), "selected next data points, seq: {}, cond: {}".format(np.array(total_seq)[sort_index], np.array(total_x)[sort_index]))

    model_logger.info("model-acq ({})".format(device), "non-selected lower_confidence_bound index, result: {}, {}".format(non_index,np.array(result)[non_index]))
    model_logger.info("model-acq ({})".format(device), "non-selected lower_confidence_bound result --> mu: {}, sigma: {}".format(np.array(selected_mu_variable)[non_index], np.array(selected_sigma_variable)[non_index]))
    model_logger.info("model-acq ({})".format(device), "non-selected next data points, seq: {}, cond: {}".format(np.array(total_seq)[non_index], np.array(total_x)[non_index]))

    return np.array(total_seq)[sort_index], np.array(total_x)[sort_index]