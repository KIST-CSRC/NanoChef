import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, argparse
import torch
from torchinfo import summary
from pyDOE import lhs
from itertools import combinations, permutations

# from olympus.noises.noise_gaussian_noise import GaussianNoise
from olympus.surfaces import SurfaceLoader
# from Sequence.SeqOpt_without_NN_gamma import SeqOpt
from Sequence.SeqOpt import SeqOpt
from Sequence.acq.acq_func_BNN_seq import acq_max_virtual
from Log.Logging_Class import ModelLogger
from Sequence.utils.functions import caculateMAE_seq, load_json_to_dict, train_seq, output_space_property

np.set_printoptions(suppress=True,threshold=np.inf, precision=6)

def main(file_path, device):
    """
    Definition of variables
    """
    # JSON 파일을 딕셔너리로 변환하여 저장
    config_dict = load_json_to_dict(file_path)

    subject=config_dict["subject"]
    log_level=config_dict["log_level"]
    model_name=config_dict["model_name"]

    total_surfaces=config_dict["total_surfaces"]

    num_variables=config_dict["num_variables"] # 변수의 개수
    initial_n_sample = config_dict["initial_n_sample"] # sampling의 개수
    n_points=config_dict["n_points"]
    batch_size=config_dict["batch_size"]
    ps_dim=config_dict["ps_dim"]
    output_dim=config_dict["output_dim"] # hyperparameter of positional encoding
    nn_n_hidden=config_dict["nn_n_hidden"] # hyperparameter
    kappa_list=config_dict["kappa_list"]
    seed_num=config_dict["seed_num"]
    np.random.seed(seed_num)

    reagent_list=config_dict["reagent_list"]
    rgn_vec_onoff=config_dict["rgn_vec_onoff"]
    
    n_search_epochs=config_dict["n_search_epochs"]
    n_train_epochs=config_dict["n_train_epochs"]
    lr=config_dict["lr"]
    patience =config_dict["patience"]   # early stopping

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    generate objects
    """
    for part_surfaces in total_surfaces:
        surfaces_name=""
        for surface_name in part_surfaces:
            surfaces_name+=surface_name
            if part_surfaces[len(part_surfaces)-1] != surface_name:
                surfaces_name+="&"
        for kappa in kappa_list:
            """
            make folder
            """
            # BASIC_PATH="/{}/{}/{}".format("data", "NFS_Data", "virtual_test")
            BASIC_PATH="{}".format("virtual_test")
            TOTAL_FOLDER = "{}/{}/{}/{}/kappa={},seed={}".format(BASIC_PATH, model_name, subject, surfaces_name, kappa, seed_num)
            if os.path.isdir(TOTAL_FOLDER) == False:
                os.makedirs(TOTAL_FOLDER)
            """
            Definition of reagent sequences

            ex)
            reagent_list=["A","B","C"]
            reagent_seq_tuple = [
                ('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), 
                ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')
            ]
            reagent_seqs = np.array([
                ['A', 'B', 'C'],
                ['A', 'C', 'B'],
                ['B', 'A', 'C'],
                ['B', 'C', 'A'],
                ['C', 'A', 'B'],
                ['C', 'B', 'A']
            ])
            """
            reagent_seq_tuple=list(permutations(reagent_list))
            reagent_seqs = np.array([np.array(t) for t in reagent_seq_tuple])
            """
            Definition for x values of virtual space 
            """
            surface_loader=SurfaceLoader()
            model_logger=ModelLogger(model_name,log_level, TOTAL_FOLDER)
            model_logger.info("config ({})".format(surfaces_name), config_dict)
            """
            generate xy points for test
            """
            domain = np.linspace(0, 1, n_points)
            X, Y = np.meshgrid(domain, domain)
            xy_points = np.c_[X.ravel(), Y.ravel()]
            test_X_points=[]
            test_reagent_seq=[]
            for reagent_seq in reagent_seqs:
                for xy_point in xy_points:
                    test_X_points.append(xy_point)
                    test_reagent_seq.append(reagent_seq)
            test_X_points=np.array(test_X_points)
            test_reagent_seq=np.array(test_reagent_seq)
            """
            generate surface --> insert in surfaces_obj_list for matching synthesis sequences
            """
            surfaces_obj_list=[]
            max_z_list=[]
            min_z_list=[]
            Z_list=[]
            Z_max_xy_point_list=[]
            Z_min_xy_point_list=[]
            for _, surface_name in enumerate(part_surfaces):
                surface_obj = surface_loader.import_surface(surface_name)()
                surfaces_obj_list.append(surface_obj)

                Z = np.zeros((len(domain), len(domain)))
                for y_index, y in enumerate(domain):
                    for x_index, x in enumerate(domain):
                        if surface_name=="HyperEllipsoid":
                            value = surface_obj.run(params=[x,y])[0][0]/20 + 1.25
                        elif surface_name=="Denali":
                            value = surface_obj.run(params=[x,y])[0][0]/2 + 4
                        elif surface_name=="MontBlanc":
                            value = surface_obj.run(params=[x,y])[0][0]/12 + 5
                        elif surface_name=="Kilimanjaro":
                            value = surface_obj.run(params=[x,y])[0][0]/4.5 + 5
                        elif surface_name=="Dejong":
                            value = surface_obj.run(params=[x,y])[0][0]
                        else:
                            raise Exception("surface_name error")
                        Z[y_index, x_index] = np.array([value]) # [ [0, 1, 2, ... ]--> 이 부분이 x 좌표에 해당함 ]
                """
                search maximum and minimum points for each surface
                """
                Z_list.append(Z)
                max_z=Z.max()
                min_z=Z.min()
                max_z_list.append(max_z)
                min_z_list.append(min_z)
                Z_max_row_index, Z_max_col_index = np.where(Z == max_z)
                Z_min_row_index, Z_min_col_index = np.where(Z == min_z)
                Z_max_xy_point=[X[Z_max_row_index[0], Z_max_col_index[0]], Y[Z_max_row_index[0], Z_max_col_index[0]]]
                Z_min_xy_point=[X[Z_min_row_index[0], Z_min_col_index[0]], Y[Z_min_row_index[0], Z_min_col_index[0]]]
                Z_max_xy_point_list.append(Z_max_xy_point)
                Z_min_xy_point_list.append(Z_min_xy_point)
                model_logger.info("surface ({})".format(surface_name), "surface={}, min_z={}, max_z={}, min_xy_points={}, max_xy_points={}".format(surface_obj.kind,min_z,max_z,Z_min_xy_point,Z_max_xy_point))
            """
            n개의 surface 중 가장 max와 min 값을 sorting
            """
            total_min_z=min(min_z_list)
            total_max_z=max(max_z_list)
            min_max=abs(total_max_z-total_min_z)
            model_logger.info("surfaces ({})".format(surfaces_name), "total_min_z({})={}, tota_max_z({})={}, min_max={}".format(part_surfaces[min_z_list.index(total_min_z)],total_min_z,part_surfaces[max_z_list.index(total_max_z)],total_max_z, min_max))
            """
            initialize target space
            """
            # LHS를 사용하여 표본 추출 --> SeqOpt에서 import할 때 이미 np.random.seed(seed)로 고정이 되어있음.
            np.random.seed(seed_num)
            x_variables = lhs(num_variables, samples=initial_n_sample*len(reagent_seqs), criterion='maximin') 
            surface_name_to_seq={}
            surface_name_to_surface_obj={}
            target_space=[]

            for i, surface_obj in enumerate(surfaces_obj_list): 
                for j in range(initial_n_sample*i,initial_n_sample*(i+1)):
                    target_info={}
                    x_variable=x_variables[j]
                    target_info["x"]=x_variables[j]
                    target_info["seq"]=reagent_seqs[i]
                    surface_name_temp=surface_obj.kind+str(i)
                    surface_name_to_seq[surface_name_temp]=reagent_seqs[i]
                    surface_name_to_surface_obj[surface_name_temp]=surface_obj

            for i, surface_obj in enumerate(surfaces_obj_list): 
                for j in range(initial_n_sample*i,initial_n_sample*(i+1)):
                    target_info={}
                    x_variable=x_variables[j]
                    target_info["x"]=x_variables[j]
                    target_info["seq"]=reagent_seqs[i]

                    for surface_name, seq in surface_name_to_seq.items():
                        result = np.array_equal(seq, reagent_seqs[i])
                        if result==True:
                            if "HyperEllipsoid" in surface_name:
                                value = (surface_obj.run(x_variable)[0][0]/20 + 1.25 -total_min_z)/min_max
                                # print("output : ", surface_obj.run(x_variable)[0][0]/20 + 1.25)
                            elif "Denali" in surface_name:
                                value = (surface_obj.run(x_variable)[0][0]/2 + 4 -total_min_z)/min_max
                                # print("output : ", surface_obj.run(x_variable)[0][0]/2 + 4)
                            elif "MontBlanc" in surface_name:
                                value = (surface_obj.run(x_variable)[0][0]/12 + 5 -total_min_z)/min_max
                            elif "Kilimanjaro" in surface_name:
                                # model_logger.info("test", "output : {}".format(surface_obj.run(x_variable)[0][0]))
                                # model_logger.info("test", "total_min_z : {}".format(total_min_z))
                                # model_logger.info("test", "converted output : {}".format(surface_obj.run(x_variable)[0][0]/4.5 +5))
                                value = (surface_obj.run(x_variable)[0][0]/4.5 +5 -total_min_z)/min_max
                                # print("output : ", surface_obj.run(x_variable)[0][0]/12 + 5)
                            elif "Dejong" in surface_name:
                                value = (surface_obj.run(x_variable)[0][0] -total_min_z)/min_max
                                # model_logger.info("test", "min_max : {}".format(min_max))
                            else:
                                raise Exception("surface_name error")
                            # print("value : ", value)
                    
                            target_info["target"]=np.array([value])
                            target_space.append(target_info)
            
            model_logger.info("target_space ({})".format(surfaces_name), "target_space_list:{}".format(target_space))
            model_logger.info("target_space ({})".format(surfaces_name), "target_space_list: len {}".format(len(target_space)))
            model_logger.info("target_space", "surface_name_to_seq: {}".format(surface_name_to_seq))
            """
            training & search 
            """
            best_mae = float('inf')
            best_y = float('inf')
            best_loss_list=[]
            best_mae_list=[]
            best_y_list=[]
            for search_epoch in range(n_search_epochs):
                """
                generate optimization object
                """
                SeqOpt_obj=SeqOpt(
                    reagent_list=reagent_list,
                    ps_dim=ps_dim,
                    output_dim=output_dim, # output dimension of positional encoding
                    num_variables=num_variables, # output dimension of conditional vector
                    nn_n_hidden=nn_n_hidden, # hyperparameter
                    seed_num=seed_num,
                    device=device,
                    rgn_vec_onoff=rgn_vec_onoff).to(device)
                SeqOpt_obj.nn_block.n_observation=len(target_space)/len(reagent_seqs)*2
                # SeqOpt_obj.bnn_block.n_observation=len(target_space)
                if search_epoch==0:
                    model_logger.info("model ({}-{}/{})".format(surfaces_name,search_epoch,n_search_epochs), "model summary: {}".format(summary(SeqOpt_obj)))
                optimizer = torch.optim.Adam(SeqOpt_obj.parameters(), lr=lr)
                """
                training
                """
                return_SeqOpt_obj, best_loss, best_model_state_dict=train_seq(surfaces_name, search_epoch, n_search_epochs, SeqOpt_obj, optimizer, model_logger, target_space, patience, n_train_epochs, device)
                torch.save(best_model_state_dict, '{}/model_stat_{}.pt'.format(TOTAL_FOLDER, search_epoch))
                best_loss_list.append(best_loss)
                y_real_pred, best_mae=caculateMAE_seq(surfaces_name, search_epoch, n_search_epochs, return_SeqOpt_obj, model_logger, target_space, device)
                best_mae_list.append(best_mae)                                                                                 
                """
                remove data points from remained data points
                """
                for target_info in target_space:
                    x_indices = np.where(np.all(test_X_points == target_info['x'], axis=1))[0]
                    seq_indices = np.where(np.all(test_reagent_seq == target_info['seq'], axis=1))[0]
                    # 동시에 맞을 때만!!!
                    common_index = list(set(x_indices) & set(seq_indices))
                    if len(common_index)==0:
                        continue
                    # delete the sampled data points
                    test_X_points=np.delete(test_X_points, common_index[0], axis=0)
                    test_reagent_seq=np.delete(test_reagent_seq, common_index[0], axis=0)
                """
                recommend next synthesis recipe with acquisition function
                """
                next_seq_list, next_rgn_list=acq_max_virtual(seq_opt_obj=return_SeqOpt_obj, model_logger=model_logger, input_seq_points=test_reagent_seq, input_X_points=test_X_points, batch_size=batch_size, kappa=kappa, device=device)
                model_logger.info("target_space ({}-{}/{})".format(surfaces_name,search_epoch,n_search_epochs), "next seq:{}, cond:{}".format(next_seq_list, next_rgn_list))
                """
                execute next synthesis recipe --> get new target_space
                """
                next_rgn_condition_list=[]
                for i in range(len(next_rgn_list)):
                    new_target_info={}
                    new_target_info["x"]=next_rgn_list[i]
                    new_target_info["seq"]=next_seq_list[i]
                    matching_surface_name = [key for key, value in surface_name_to_seq.items() if np.array_equal(value,next_seq_list[i])][0]

                    for surface_name, seq in surface_name_to_seq.items():
                        result = np.array_equal(seq, next_seq_list[i])
                        if result==True:
                            if "HyperEllipsoid" in surface_name:
                                value = (surface_name_to_surface_obj[matching_surface_name].run(next_rgn_list[i])[0][0]/20 + 1.25 -total_min_z)/min_max
                            elif "Denali" in surface_name:
                                value = (surface_name_to_surface_obj[matching_surface_name].run(next_rgn_list[i])[0][0]/2 + 4 -total_min_z)/min_max
                            elif "MontBlanc" in surface_name:
                                value = (surface_name_to_surface_obj[matching_surface_name].run(next_rgn_list[i])[0][0]/12 + 5 -total_min_z)/min_max
                            elif "Kilimanjaro" in surface_name:
                                # model_logger.info("test", "output : {}".format(surface_name_to_surface_obj[matching_surface_name].run(next_rgn_list[i])[0][0]))
                                value = (surface_name_to_surface_obj[matching_surface_name].run(next_rgn_list[i])[0][0]/4.5 +5 -total_min_z)/min_max
                            elif "Dejong" in surface_name:
                                value = (surface_name_to_surface_obj[matching_surface_name].run(next_rgn_list[i])[0][0] -total_min_z)/min_max
                                # model_logger.info("test", "total_min_z : {}".format(total_min_z))
                                # model_logger.info("test", "min_max : {}".format(min_max))
                            else:
                                raise Exception("surface_name error")
                    new_target_info["target"]=np.array([value])
                    # new_target_info["surface"]=surface.kind
                    target_space.append(new_target_info)
                if (search_epoch%10)+1 == 10:
                    model_logger.info("target_space ({}-{}/{})".format(surfaces_name,search_epoch+1,n_search_epochs), "target_space after experiments : {}".format(target_space))
                """
                target space 분리
                """
                initial_sampling_target_space=target_space[:initial_n_sample*len(reagent_seqs)]
                ai_target_space=target_space[initial_n_sample*len(reagent_seqs):]
                previous_epoch_ai_target_space=ai_target_space[:(search_epoch)*batch_size]
                current_epoch_ai_target_space=ai_target_space[search_epoch*batch_size:(search_epoch+1)*batch_size]

                # initial_sampling_target_space=target_space[:initial_n_sample]
                # ai_target_space=target_space[initial_n_sample:]
                # previous_epoch_ai_target_space=ai_target_space[:(search_epoch)*batch_size]
                # current_epoch_ai_target_space=ai_target_space[search_epoch*batch_size:(search_epoch+1)*batch_size]

                initial_surface_X_list=[[] for _ in range(len(reagent_seqs))]
                initial_surface_Y_list=[[] for _ in range(len(reagent_seqs))]

                previous_ai_surface_X_list=[[] for _ in range(len(reagent_seqs))]
                previous_ai_surface_Y_list=[[] for _ in range(len(reagent_seqs))]

                current_ai_surface_X_list=[[] for _ in range(len(reagent_seqs))]
                current_ai_surface_Y_list=[[] for _ in range(len(reagent_seqs))]

                for target_info in initial_sampling_target_space:
                    for i in range(len(reagent_seqs)):
                        if np.array_equal(target_info['seq'],np.array(reagent_seqs[i])):
                            initial_surface_X_list[i].append(target_info['x'][0])
                            initial_surface_Y_list[i].append(target_info['x'][1])
                for target_info in previous_epoch_ai_target_space:
                    for i in range(len(reagent_seqs)):
                        if np.array_equal(target_info['seq'],np.array(reagent_seqs[i])):
                            previous_ai_surface_X_list[i].append(target_info['x'][0])
                            previous_ai_surface_Y_list[i].append(target_info['x'][1])
                for target_info in current_epoch_ai_target_space:
                    for i in range(len(reagent_seqs)):
                        if np.array_equal(target_info['seq'],np.array(reagent_seqs[i])):
                            current_ai_surface_X_list[i].append(target_info['x'][0])
                            current_ai_surface_Y_list[i].append(target_info['x'][1])
                """
                draw scatter plot of surface 0
                """
                for i in range(len(reagent_seqs)):
                    fig = plt.figure()
                    ax = plt.axes()
                    levels = np.linspace(total_min_z, total_max_z, 200)

                    pc0 = plt.contourf(X, Y, Z_list[i], cmap = "coolwarm", levels=levels)
                    cbar0 = plt.colorbar(pc0)

                    plt.scatter(Z_min_xy_point_list[i][0], Z_min_xy_point_list[i][1], marker='*', c="Yellow", s=75, label='global')
                    plt.scatter(initial_surface_X_list[i], initial_surface_Y_list[i], marker='D', c="grey", alpha=0.8, label='initial')
                    plt.scatter(previous_ai_surface_X_list[i], previous_ai_surface_Y_list[i], marker='o', c="grey", alpha=0.8, label='previous')
                    plt.scatter(current_ai_surface_X_list[i], current_ai_surface_Y_list[i], marker='v', c="purple", alpha=1, label='current')
                    plt.tight_layout()
                    plt.savefig('{}/scattertest_{}_{}.png'.format(TOTAL_FOLDER, surfaces_obj_list[i].kind+str(i), search_epoch),dpi=300)
                    plt.close()

                """
                draw y_values scatter plot
                """
                x_values=[]
                initial_n_cycles=int(initial_n_sample/batch_size)
                for i in range(initial_n_cycles):
                    for j in range(batch_size*len(reagent_seqs)):
                        x_values.append(i)
                for i in range(initial_n_cycles, initial_n_cycles+search_epoch+1):
                    for j in range(batch_size):
                        x_values.append(i)
                y_values=[target_info["target"] for target_info in target_space]
                # print("len(x_values), x_values", len(x_values), x_values)
                # print("len(y_values), y_values", len(y_values), y_values)
                plt.scatter(x_values, y_values, label='y_searched')
                plt.xlabel('Search epochs')
                plt.ylabel('y_searched')
                plt.title("{}_{}".format(subject, surfaces_name))
                plt.tight_layout()
                plt.savefig('{}/result_searched_y_{}.png'.format(TOTAL_FOLDER,surfaces_name),dpi=300)
                plt.close()
                # CSV 파일로 저장
                df = pd.DataFrame({'Search epochs': x_values, 'y_searched': y_values})
                df.to_csv('{}/result_searched_y_{}.csv'.format(TOTAL_FOLDER,surfaces_name), index=False)
                """
                draw best_y_values csv
                """
                best_y=min(y_values)
                best_y_list.append(best_y)
                # CSV 파일로 저장
                df = pd.DataFrame({'Search epochs': [x for x in range(len(best_y_list))], 'y_searched': best_y_list})
                df.to_csv('{}/result_searched_best_y_{}.csv'.format(TOTAL_FOLDER,surfaces_name), index=False)
                """
                draw loss scatter plot
                """
                x_values_loss_mae=[i for i in range(initial_n_cycles, initial_n_cycles+search_epoch+1)]
                plt.scatter(x_values_loss_mae, best_loss_list, label='Loss')
                plt.xlabel('Search epochs')
                plt.ylabel('Loss')
                plt.title("{}_{}".format(subject, surfaces_name))
                plt.legend()
                plt.tight_layout()
                plt.savefig('{}/result_loss_{}.png'.format(TOTAL_FOLDER,surfaces_name),dpi=300)
                plt.close()
                # CSV 파일로 저장
                df = pd.DataFrame({'Search epochs': x_values_loss_mae, 'Loss': best_loss_list})
                df.to_csv('{}/result_loss_{}.csv'.format(TOTAL_FOLDER,surfaces_name), index=False)
                """
                draw mae scatter plot
                """
                plt.scatter(x_values_loss_mae, best_mae_list, label='MAE')
                plt.xlabel('Search epochs')
                plt.ylabel('MAE')
                plt.title("{}_{}".format(subject, surfaces_name))
                plt.legend()
                plt.tight_layout()
                plt.savefig('{}/result_mae_{}.png'.format(TOTAL_FOLDER,surfaces_name),dpi=300)
                plt.close()
                # CSV 파일로 저장
                df = pd.DataFrame({'Search epochs': x_values_loss_mae, 'MAE': best_mae_list})
                df.to_csv('{}/result_mae_{}.csv'.format(TOTAL_FOLDER,surfaces_name), index=False)

                # 데이터 분리
                y_real = [d[0] for d in y_real_pred]
                y_pred = [d[1] for d in y_real_pred]
                # y=x 그래프 추가
                # scatter plot 그리기
                plt.figure(figsize=(8, 6))
                plt.scatter(y_real, y_pred, color='blue')
                plt.plot(y_real, y_real, color='red', linestyle='--', label='y=x')
                plt.xlabel('y_real')
                plt.ylabel('y_pred')
                plt.title("y_real vs y_pred")
                plt.grid(True)
                plt.savefig('{}/result_y_real_vs_y_pred_{}.png'.format(TOTAL_FOLDER, surfaces_name),dpi=300)
                plt.close()
                # CSV 파일로 저장
                df = pd.DataFrame({'y_real': y_real, 'y_pred': y_pred})
                df.to_csv('{}/result_y_real_vs_y_pred_{}.csv'.format(TOTAL_FOLDER, surfaces_name), index=False)

                # extract target space
                output_space_property(target_space=target_space, dirname=TOTAL_FOLDER, filename="result_{}".format(surfaces_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--path", help="config file path")
    parser.add_argument("--cuda", help="cuda device ('cpu' or 'cuda:0' or 'cuda:1'... )")
    
    args = parser.parse_args()
    main(args.path, args.cuda)

    # python surface_test_NN_seq_multi_2.py --path Sequence/config/20240814/NN_Gamma_seq_dejong1.json --cuda cuda:0
    # python surface_test_NN_seq_multi_2.py --path Sequence/config/20240814/NN_Gamma_seq_dejong2.json --cuda cuda:1
