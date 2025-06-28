import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualization_model_performance(dir_paths, color_list, kappa_list, seed_list, root_path):

    for color_i, dir_path in enumerate(dir_paths):
        for kappa in kappa_list:
            for seed in seed_list:
                data = pd.read_csv("{}/{}/kappa={},seed={}/result_{}.csv".format(root_path, dir_path, kappa, seed, dir_path))
                best_data = pd.read_csv("{}/{}/kappa={},seed={}/result_searched_best_y_{}.csv".format(root_path, dir_path, kappa, seed, dir_path))

                # DataFrame 생성
                df = pd.DataFrame(data)
                best_df = pd.DataFrame(best_data)

                x_data = np.array([np.fromstring(data_str.strip('[]'), sep=' ') for data_str in df['x']])
                seq_data = np.array([np.array([elem.strip("'") for elem in seq.strip("[]").split()]) for seq in df['seq']])
                target_data = np.array([np.array([float(data_str.strip('[]'))]) for data_str in df['target']])

                best_target_data = np.array([np.array([float(data_str.strip('[]'))]) for data_str in best_df['y_searched']])

                x_data_sampling=x_data[:40]
                seq_data_sampling=seq_data[:40]
                target_data_sampling=target_data[:40]

                x_data_ai=x_data[40:]
                seq_data_ai=seq_data[40:]
                target_data_ai=target_data[40:]

                seq_a=[]
                seq_b=[]
                seq_a_num=[]
                seq_b_num=[]
                seq_a_num_accumulated=[]
                seq_b_num_accumulated=[]

                y_value_a=[]
                y_index_a=[]
                y_value_b=[]
                y_index_b=[]

                for i in range(100):
                    seq_cycle_datas=seq_data_ai[4*i:4*i+4]
                    target_cycle_datas=target_data_ai[4*i:4*i+4]
                    seq_a_=0
                    seq_b_=0
                    for idx, seq_cycle_data in enumerate(seq_cycle_datas):
                        # print(type(seq_cycle_data[0]))
                        # print(seq_cycle_data[0])
                        if seq_cycle_data[0] == "AgNO3":
                            seq_a.append(seq_cycle_data)
                            seq_a_+=1
                            y_value_a.append(target_cycle_datas[idx])
                            y_index_a.append(i)
                        elif seq_cycle_data[0] == "NaBH4":
                            seq_b.append(seq_cycle_data)
                            seq_b_+=1
                            y_value_b.append(target_cycle_datas[idx])
                            y_index_b.append(i)
                    seq_a_num.append(seq_a_)
                    seq_b_num.append(seq_b_)
                    seq_a_num_accumulated.append(len(seq_a))
                    seq_b_num_accumulated.append(len(seq_b))

                # 라인 플롯 그리기
                f, ax = plt.subplots(figsize=(6.5, 5))
                x=[i for i in range(100)]
                # surface_a, surface_b=dir_path.split("&")
                surface_a="Dejong1"
                surface_b="Dejong2"
                ax.scatter(x, seq_a_num, s=8, label=surface_a, color="#B4685D")
                ax.scatter(x, seq_b_num, s=8, label=surface_b, color=color_list[color_i],marker='v')
                # ax.step(x, best_target_data, where='post', label=surface_a, color="g")

                ax.spines['left'].set_linewidth(4) # 선 두께
                ax.spines['right'].set_linewidth(4) # 선 두께
                ax.spines['top'].set_linewidth(4) # 선 두께
                ax.spines['bottom'].set_linewidth(4) # 선 두께

                ax.spines['left'].set_color("#5A5F5E") # 선 두께
                ax.spines['right'].set_color("#5A5F5E") # 선 두께
                ax.spines['top'].set_color("#5A5F5E") # 선 두께
                ax.spines['bottom'].set_color("#5A5F5E") # 선 두께
                
                ax.tick_params(direction="in", width=3, color="#5A5F5E", length=8, pad=12, labelcolor="#5A5F5E")
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"
                plt.xticks(fontsize=8, fontname="Arial")
                plt.yticks(fontsize=8, fontname="Arial")
                # 그래프 제목 및 축 레이블 설정
                # plt.title('Seach counts')
                plt.ylim(-0.99, 4.99)
                plt.xlabel('Search epochs')
                plt.ylabel('Counts')
                # plt.legend()
                plt.tight_layout()
                
                plt.savefig("{},kappa={},seed={}_epoch_counts.png".format(dir_path, kappa, seed), dpi=300)
                plt.close()


                # 라인 플롯 그리기
                f, ax = plt.subplots(figsize=(6.5, 5))
                x=[i for i in range(100)]
                surface_a, surface_b=dir_path.split("&")
                ax.step(x, seq_a_num_accumulated, where='post', label=surface_a, color="#B4685D", linewidth=2)
                ax.step(x, seq_b_num_accumulated, where='post', label=surface_a, color=color_list[color_i], linewidth=2)
                # ax.scatter(x, seq_a_num_accumulated, s=8, label=surface_a, color="#B4685D")
                # ax.scatter(x, seq_b_num_accumulated, s=8, label=surface_b, color=color_list[color_i],marker='v')

                ax.spines['left'].set_linewidth(4) # 선 두께
                ax.spines['right'].set_linewidth(4) # 선 두께
                ax.spines['top'].set_linewidth(4) # 선 두께
                ax.spines['bottom'].set_linewidth(4) # 선 두께

                ax.spines['left'].set_color("#5A5F5E") # 선 두께
                ax.spines['right'].set_color("#5A5F5E") # 선 두께
                ax.spines['top'].set_color("#5A5F5E") # 선 두께
                ax.spines['bottom'].set_color("#5A5F5E") # 선 두께

                ax.tick_params(direction="in", width=3, color="#5A5F5E", length=8, pad=12, labelcolor="#5A5F5E")
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"
                plt.xticks(fontsize=8, fontname="Arial")
                plt.yticks(fontsize=8, fontname="Arial")
                # 그래프 제목 및 축 레이블 설정
                # plt.title('Seach counts')
                plt.ylim(-25, 425)
                plt.xlabel('Search epochs')
                plt.ylabel('Counts')
                plt.tight_layout()
                # plt.legend()
                plt.savefig("{},kappa={},seed={}_accumulated.png".format(dir_path, kappa, seed), dpi=300)
                plt.close()


                # 라인 플롯 그리기
                f, ax = plt.subplots(figsize=(6.5, 5))
                ax.scatter(y_index_a, y_value_a, s=6, label=surface_a, color="#B4685D")
                ax.scatter(y_index_b, y_value_b, s=6, label=surface_b, color=color_list[color_i],marker='v')
                ax.step(x, best_target_data, where='post', label=surface_a, color="#548235", alpha=0.6, linewidth=2)

                ax.spines['left'].set_linewidth(4) # 선 두께
                ax.spines['right'].set_linewidth(4) # 선 두께
                ax.spines['top'].set_linewidth(4) # 선 두께
                ax.spines['bottom'].set_linewidth(4) # 선 두께

                ax.spines['left'].set_color("#5A5F5E") # 선 두께
                ax.spines['right'].set_color("#5A5F5E") # 선 두께
                ax.spines['top'].set_color("#5A5F5E") # 선 두께
                ax.spines['bottom'].set_color("#5A5F5E") # 선 두께

                ax.tick_params(direction="in", width=3, color="#5A5F5E", length=8, pad=12, labelcolor="#5A5F5E")
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"
                plt.xticks(fontsize=8, fontname="Arial")
                plt.yticks(fontsize=8, fontname="Arial")
                # 그래프 제목 및 축 레이블 설정
                # plt.title('Result of AI-decision process')
                plt.xlim(-10, 105)
                plt.ylim(-0.1, 1.1)
                plt.xlabel('Search epochs')
                plt.ylabel('Value')
                plt.tight_layout()
                # plt.legend()
                plt.savefig("{},kappa={},seed={}_y_value.png".format(dir_path, kappa, seed), dpi=300)
                plt.close()

def visualization_scatter(cycle, dirname, filename, reagent_seqs, batch_size, initial_n_sample, savefilename_list):
    
    target_space=pd.read_csv(dirname+filename)
    for search_epoch in range(cycle):
        initial_sampling_target_space=target_space.iloc[:initial_n_sample*len(reagent_seqs)].values
        ai_target_space=target_space.iloc[initial_n_sample*len(reagent_seqs):].values
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
                if reagent_seqs[i] in target_info:
                    initial_surface_X_list[i].append(target_info[0])
                    initial_surface_Y_list[i].append(target_info[1])
        for target_info in previous_epoch_ai_target_space:
            for i in range(len(reagent_seqs)):
                if reagent_seqs[i] in target_info:
                    previous_ai_surface_X_list[i].append(target_info[0])
                    previous_ai_surface_Y_list[i].append(target_info[1])
        for target_info in current_epoch_ai_target_space:
            for i in range(len(reagent_seqs)):
                if reagent_seqs[i] in target_info:
                    current_ai_surface_X_list[i].append(target_info[0])
                    current_ai_surface_Y_list[i].append(target_info[1])
        """
        draw scatter plot of surface 0
        """
        for i in range(len(reagent_seqs)):
            fig = plt.figure()
            ax = plt.axes()
            # levels = np.linspace(total_min_z, total_max_z, 200)

            # pc0 = plt.contourf(X, Y, Z_list[i], cmap = "coolwarm", levels=levels)
            # cbar0 = plt.colorbar(pc0)

            # plt.scatter(Z_min_xy_point_list[i][0], Z_min_xy_point_list[i][1], marker='*', c="Yellow", s=75, label='global')
            plt.scatter(initial_surface_X_list[i], initial_surface_Y_list[i], marker='D', c="grey", alpha=0.8, label='initial')
            plt.scatter(previous_ai_surface_X_list[i], previous_ai_surface_Y_list[i], marker='o', c="grey", alpha=0.8, label='previous')
            plt.scatter(current_ai_surface_X_list[i], current_ai_surface_Y_list[i], marker='v', c="purple", alpha=1, label='current')
            plt.tight_layout()
            plt.savefig('{}/scattertest_{}_{}.png'.format(dirname, savefilename_list[i], search_epoch),dpi=300)
            plt.close()

def create_gif(images, gif_path, duration=500, loop=0):
    """
    images: 이미지 파일 경로 리스트
    gif_path: 저장할 GIF 파일 경로
    duration: 각 이미지의 표시 시간 (밀리초)
    loop: 애니메이션 반복 횟수 (0은 무한 반복)
    """
    frames = [Image.open(image) for image in images]

    # GIF 생성
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

def visualization_gif(surface_list, seed_list, kappa_list, root_dirpath):
    for idx, surfaces in enumerate(surface_list):
        for seed in seed_list:
                for kappa in kappa_list:
                    dir_path="{}/{}/kappa={},seed={}".format(root_dirpath, surfaces, kappa, seed)
                    if surfaces == "DejongDejong":
                        surface_list=["Dejong", "Dejong"]
                    else:
                        pre_surface_list = surfaces.split("&")
                        surface_list=[]
                        for surface in pre_surface_list:
                            if surface =="KilimanjaroKilimanjaro":
                                surface_list.append("Kilimanjaro")
                                surface_list.append("Kilimanjaro")
                            elif surface =="DenaliDenali":
                                surface_list.append("Denali")
                                surface_list.append("Denali")
                            else:
                                surface_list.append(surface)
                                
                    for idx, surface in enumerate(surface_list):
                        image_paths=["{}/scattertest_{}{}_{}.png".format(dir_path, surface, idx, i) for i in range(100)]

                        # 생성할 GIF 파일 경로
                        output_gif_path = '{}/output_{}_{}.gif'.format(dir_path, surface, idx)

                        # create_gif 함수 호출
                        
                        
                        create_gif(image_paths, output_gif_path)
                        print("GIF Generation!!:{}".format(output_gif_path))


if __name__ == "__main__":
    dir_paths=[
        # "Dejong&Dejong",
        # "Dejong&HyperEllipsoid",
        # "Dejong&Denali",
        # "Dejong&MontBlanc",
        "Dejong&Kilimanjaro"
    ]

    color_list=[
        # "#B4685D", # 빨강
        "#7A6991", # 보라
        "#6674A1", # 파랑
        "#698b89", # 연초
        "#c79236", # 찐노
        # "#ed7e6f"  # 연분홍
    ]
    kappa_list=[
        # 0.05,
        0.1,
        0.25,
        # 0.5,
    ]
    seed_list=[
        1,2,3,4,5
    ]
    root_path="virtual_test/Evidence+SEQ/Evidence MCMC100+layer3_hidden500+ReLU"

    visualization_model_performance(dir_paths, color_list, kappa_list, seed_list, root_path)

    cycle=25
    dirname="Data/SeqOpt-3000_513_var2/Result/real/20241006/"
    filename="20241006_141609_25_data_real.csv"
    reagent_seqs=[
        "['AgNO3', 'NaBH4']",
        "['NaBH4', 'AgNO3']"
    ]
    batch_size=4
    initial_n_sample=10
    savefilename_list=[
        "MR","RM"
    ]
    
    visualization_scatter(cycle,
        dirname,
        filename,
        reagent_seqs,
        batch_size,
        initial_n_sample,
        savefilename_list)
    

    # 이미지 파일 경로 리스트 (합성하고 싶은 이미지들)
    surface_list=[
    #     "Dejong&Dejong&Denali&Denali&KilimanjaroKilimanjaro",
    #     "Dejong&Dejong&HyperEllipsoid&HyperEllipsoid&DenaliDenali",
    #     "Dejong&Dejong&HyperEllipsoid&HyperEllipsoid&KilimanjaroKilimanjaro",
    #     "Denali&Denali&HyperEllipsoid&HyperEllipsoid&KilimanjaroKilimanjaro",
        "DejongDejong",
        "Dejong&HyperEllipsoid",
        # "Dejong&Denali",
        # "Dejong&Kilimanjaro",
    ]

    seed_list=[
        1,
        # 2,
        3,
        4,
        # 5,
    ]

    kappa_list=[
        # 0.01,
        # 0.05,
        # 0.1,
        # 0.25,
        # 0.5
        1,2,3,4,5
    ]

    root_dirpath="virtual_test/Evidence+SEQ/Evidence MCMC100+layer3_hidden500+ReLU"

    visualization_gif(surface_list,
    seed_list,
    kappa_list,
    root_dirpath)