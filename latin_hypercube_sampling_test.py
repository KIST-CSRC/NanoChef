from scipy.stats import qmc
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
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

for i in range(4):
    np.random.seed(1)
    sample = latin_hypercube_sampling(2, 100, seed=i)  # 재현성을 위해 seed 설정

    # 그래프 생성
    plt.figure(figsize=(6, 6))
    plt.scatter(sample[:, 0], sample[:, 1], c='b', alpha=0.6, edgecolors='k')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Latin Hypercube Sampling {i+1}')
    plt.grid(True)

    # 파일 저장
    filename = f"lhs_sample=100_{i+1}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 메모리 해제

    print(f"Saved: {filename}")