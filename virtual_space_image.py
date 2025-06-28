import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from itertools import combinations

from olympus.noises.noise_gaussian_noise import GaussianNoise
from olympus.surfaces import SurfaceLoader

surface_loader=SurfaceLoader()

# noise=GaussianNoise(scale=0.5)
# surface=Dejong(param_dim=2, noise=noise)
# print(surface.run([[0.5, 0.5], [1, 1]]))
# print(surface.run([1, 1]))
# print(surface.kind)

# surfaces = [
#     "Dejong", "HyperEllipsoid", "Zakharov",
#     "AckleyPath", "Branin", "Levy", "Michalewicz", "Rastrigin", "Rosenbrock", "Schwefel", "StyblinskiTang",
#     "LinearFunnel", "NarrowFunnel", 
#     "GaussianMixture", "Denali", "Everest", "K2", "Kilimanjaro", "Matterhorn", "MontBlanc"
#     ]

surfaces = [
    "Dejong", "HyperEllipsoid", "AckleyPath", "Rastrigin", "LinearFunnel", "Levy",
    "Branin", "Michalewicz", "Rosenbrock", "Schwefel", "StyblinskiTang", "GaussianMixture", 
    "Denali", "Everest", "K2", "Kilimanjaro", "Matterhorn", "MontBlanc"
    ]

# for surface_index, surface_name in enumerate(surfaces):

#     surface = surface_loader.import_surface(surface_name)()

#     domain = np.linspace(0, 1, 100)
#     X, Y = np.meshgrid(domain, domain)
#     Z = np.zeros((len(domain), len(domain)))
#     for x_index, x in enumerate(domain):
#         for y_index, y in enumerate(domain):
#             value = surface.run(params=[x, y])
#             Z[x_index, y_index] = value[0][0]
#     max_z=Z.max()
#     min_z=Z.min()
#     # syntax for 3-D projection
#     fig = plt.figure()
#     ax = plt.axes(projection ='3d')

#     levels = np.linspace(min_z, max_z, 200)
#     pc = ax.contourf(X, Y, Z, cmap = "coolwarm", levels=levels)
#     cbar = plt.colorbar(pc)

#     plt.tight_layout()
#     plt.savefig('surface_{}.png'.format(surface_name),dpi=300)
#     plt.close()

selected_surfaces_list=list(combinations(surfaces, 2))
print(selected_surfaces_list)
# print(len(surfaces))
# print(selected_surfaces_list)
# print(len(selected_surfaces_list))

def calculate_spearman_rank_correlation(data1, data2):
    # 데이터를 순위로 변환
    data1=data1.ravel()
    data2=data2.ravel()
    rank_data1 = np.argsort(np.argsort(data1))
    rank_data2 = np.argsort(np.argsort(data2))

    # 중복된 원소 찾기
    unique_elements_1, counts_1 = np.unique(data1, return_counts=True)
    unique_elements_2, counts_2 = np.unique(data2, return_counts=True)

    # 중복된 원소의 평균 순위 계산
    average_ranks_1 = {element: np.mean(rank_data1[data1 == element]) for element in unique_elements_1}
    average_ranks_2 = {element: np.mean(rank_data2[data2 == element]) for element in unique_elements_2}

    # 평균 순위를 사용하여 covert_rank_arr 생성
    covert_rank_arr_1 = np.array([average_ranks_1[element] for element in data1])
    covert_rank_arr_2 = np.array([average_ranks_2[element] for element in data2])

    # 스피어맨 순위 상관 계수 계산
    spearman_corr, _ = scipy.stats.spearmanr(covert_rank_arr_1, covert_rank_arr_2)

    return spearman_corr

combination_name_list=[]
retrun_spearman_corr_list=[]

Z_0_max_x_point_list=[]
Z_0_max_y_point_list=[]
Z_0_min_x_point_list=[]
Z_0_min_y_point_list=[]
Z_1_max_x_point_list=[]
Z_1_max_y_point_list=[]
Z_1_min_x_point_list=[]
Z_1_min_y_point_list=[]

max_z_0_list=[]
min_z_0_list=[]
max_z_1_list=[]
min_z_1_list=[]

for combination in selected_surfaces_list:

    surface_0 = surface_loader.import_surface(combination[0])()
    surface_1 = surface_loader.import_surface(combination[1])()

    domain = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(domain, domain)
    Z_0 = np.zeros((len(domain), len(domain)))
    Z_1 = np.zeros((len(domain), len(domain)))
    for x_index, x in enumerate(domain):
        for y_index, y in enumerate(domain):
            value_0 = surface_0.run(params=[x, y])
            value_1 = surface_1.run(params=[x, y])
            Z_0[x_index, y_index] = value_0[0][0]
            Z_1[x_index, y_index] = value_1[0][0]
    
    combination_name="{}/{}".format(surface_0.kind, surface_1.kind)
    retrun_spearman_corr = calculate_spearman_rank_correlation(Z_0, Z_1)

    max_z_0=Z_0.max()
    min_z_0=Z_0.min()
    max_z_1=Z_1.max()
    min_z_1=Z_1.min()

    # max나 min이 중복되면 여러개 뱉어냄
    Z_0_max_row_index, Z_0_max_col_index = np.where(Z_0 == max_z_0)
    Z_0_min_row_index, Z_0_min_col_index = np.where(Z_0 == min_z_0)
    Z_1_max_row_index, Z_1_max_col_index = np.where(Z_1 == max_z_1)
    Z_1_min_row_index, Z_1_min_col_index = np.where(Z_1 == min_z_1)

    Z_0_max_point=[X[Z_0_max_row_index, Z_0_max_col_index], Y[Z_0_max_row_index, Z_0_max_col_index]]
    Z_0_min_point=[X[Z_0_min_row_index, Z_0_min_col_index], Y[Z_0_min_row_index, Z_0_min_col_index]]
    Z_1_max_point=[X[Z_1_max_row_index, Z_1_max_col_index], Y[Z_1_max_row_index, Z_1_max_col_index]]
    Z_1_min_point=[X[Z_1_min_row_index, Z_1_min_col_index], Y[Z_1_min_row_index, Z_1_min_col_index]]

    combination_name_list.append(combination_name)
    retrun_spearman_corr_list.append(retrun_spearman_corr)
    Z_0_max_x_point_list.append(X[Z_0_max_row_index, Z_0_max_col_index])
    Z_0_max_y_point_list.append(Y[Z_0_max_row_index, Z_0_max_col_index])
    Z_0_min_x_point_list.append(X[Z_0_min_row_index, Z_0_min_col_index])
    Z_0_min_y_point_list.append(Y[Z_0_min_row_index, Z_0_min_col_index])
    Z_1_max_x_point_list.append(X[Z_1_max_row_index, Z_1_max_col_index])
    Z_1_max_y_point_list.append(Y[Z_1_max_row_index, Z_1_max_col_index])
    Z_1_min_x_point_list.append(X[Z_1_min_row_index, Z_1_min_col_index])
    Z_1_min_y_point_list.append(Y[Z_1_min_row_index, Z_1_min_col_index])
    max_z_0_list.append(max_z_0)
    min_z_0_list.append(min_z_0)
    max_z_1_list.append(max_z_1)
    min_z_1_list.append(min_z_1)

# 예제 DataFrame 생성
data = {
    'Combination': combination_name_list,
    'spearman_corr': retrun_spearman_corr_list,
    "Z_0_max_x_point": Z_0_max_x_point_list,
    "Z_0_max_y_point": Z_0_max_y_point_list,
    "Z_0_min_x_point": Z_0_min_x_point_list,
    "Z_0_min_y_point": Z_0_min_y_point_list,
    "Z_1_max_x_point": Z_1_max_x_point_list,
    "Z_1_max_y_point": Z_1_max_y_point_list,
    "Z_1_min_x_point": Z_1_min_x_point_list,
    "Z_1_min_y_point": Z_1_min_y_point_list,
    "max_z_0": max_z_0_list,
    "min_z_0": min_z_0_list,
    "max_z_1": max_z_1_list,
    "min_z_1": min_z_1_list
}

df = pd.DataFrame(data)

# Excel 파일로 저장
df.to_csv('example_mountain.csv', index=False)
