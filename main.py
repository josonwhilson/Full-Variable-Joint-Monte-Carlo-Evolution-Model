# -*- coding: utf-8 -*-
"""
Model 4: Alpha-Omega Dual-Factor Kinetic Simulation with Inert Boundaries
This script simulates stochastic polymer reactions utilizing empirical GPC data 
for precise molecular weight (chain length) distribution mapping.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
from scipy.stats import norm
from numba import jit, prange
from dotenv import load_dotenv

# =============================================================================
# 1. 载入环境配置 (Load Configuration)
# =============================================================================
load_dotenv()

# Chain Configurations
MW_MODE = os.getenv('MW_MODE', 'GPC').strip().upper()
GPC_FILE_PATH = os.getenv('GPC_FILE_PATH', 'GPC_articaluse.xlsx')
MONOMER_MASS = float(os.getenv('MONOMER_MASS', 44.0))

N_SITES_FIXED = int(os.getenv('N_SITES_FIXED', 2022))
MW_MEAN = float(os.getenv('MW_MEAN', 2000.0))
MW_PDI = float(os.getenv('MW_PDI', 1.2))

# Simulation Parameters
NUM_SIMULATIONS = int(os.getenv('NUM_SIMULATIONS', 10000))
INERT_FRACTION = float(os.getenv('INERT_FRACTION', 0.01978))
ACTIVATION_PROBABILITY = float(os.getenv('ACTIVATION_PROBABILITY', 0.6))

# Dual-Factor Weights
INITIAL_ALPHA = float(os.getenv('INITIAL_ALPHA', 1.0))
INITIAL_OMEGA = float(os.getenv('INITIAL_OMEGA', 1.0))
ISOLATED_ACTIVITY = float(os.getenv('ISOLATED_ACTIVITY', 0.0))
ALPHA_DECREASE = float(os.getenv('ALPHA_DECREASE', 0.810827897))
OMEGA_DECREASE = float(os.getenv('OMEGA_DECREASE', 0.141465204))

# States
UNREACTED, REACTED, ISOLATED, INERT = 0, 2, 3, 4
MODEL_NAME = f"Model 4 Alpha-Omega ({MW_MODE} Mode)"

# =============================================================================
# 2. 核心计算引擎 (Numba 并行加速)
# =============================================================================
# （注意：这里的 run_single_simulation 和 run_parallel_simulations 保持和上一次完全一致）

@jit(nopython=True)
def run_single_simulation(n_sites, num_inert_groups, activation_prob, alpha_dec, omega_dec, init_alpha, init_omega, iso_activity):
    chain = np.full(n_sites, UNREACTED, dtype=np.int32)
    alpha_factors = np.full(n_sites, init_alpha, dtype=np.float64)
    omega_factors = np.full(n_sites, init_omega, dtype=np.float64)
    is_available = np.ones(n_sites, dtype=np.bool_)
    
    # 1. Inert Groups Initialization
    if num_inert_groups > 0:
        pool = np.arange(n_sites)
        for i in range(num_inert_groups):
            rand_idx = np.random.randint(i, n_sites)
            chosen_site = pool[rand_idx]
            pool[rand_idx] = pool[i] 
            chain[chosen_site] = INERT
            alpha_factors[chosen_site] = 0.0
            omega_factors[chosen_site] = 0.0
            is_available[chosen_site] = False

        for i in range(n_sites):
            if chain[i] == INERT:
                if i - 1 >= 0 and is_available[i - 1]:
                    alpha_factors[i - 1] = max(0.0, alpha_factors[i - 1] - alpha_dec)
                    omega_factors[i - 1] = max(0.0, omega_factors[i - 1] - omega_dec)
                if i + 1 < n_sites and is_available[i + 1]:
                    alpha_factors[i + 1] = max(0.0, alpha_factors[i + 1] - alpha_dec)
                    omega_factors[i + 1] = max(0.0, omega_factors[i + 1] - omega_dec)

    # 2. Initial Isolation Screening
    num_available = n_sites - num_inert_groups
    for i in range(n_sites):
        if is_available[i]:
            left_blocked = (i - 1 < 0) or not is_available[i - 1]
            right_blocked = (i + 1 >= n_sites) or not is_available[i + 1]
            if left_blocked and right_blocked:
                chain[i] = ISOLATED
                alpha_factors[i] = iso_activity
                omega_factors[i] = iso_activity
                is_available[i] = False
                num_available -= 1

    total_collisions = 0

    # 3. Reaction Main Loop
    while num_available > 0:
        available_indices = np.where(is_available)[0]
        if len(available_indices) == 0: break

        total_alpha = 0.0
        for idx in available_indices: total_alpha += alpha_factors[idx]

        if total_alpha <= 1e-9:
            for idx in available_indices:
                chain[idx] = ISOLATED
                is_available[idx] = False
                num_available -= 1
            break

        r = np.random.rand() * total_alpha
        cumulative_alpha = 0.0
        initial_reaction_idx = available_indices[-1] 
        for idx in available_indices:
            cumulative_alpha += alpha_factors[idx]
            if cumulative_alpha >= r:
                initial_reaction_idx = idx
                break

        total_collisions += 1
        if np.random.rand() <= activation_prob:
            left_n = initial_reaction_idx - 1
            right_n = initial_reaction_idx + 1
            left_can_react = (left_n >= 0 and is_available[left_n] and chain[left_n] == UNREACTED)
            right_can_react = (right_n < n_sites and is_available[right_n] and chain[right_n] == UNREACTED)

            site1, site2, chosen_n = -1, -1, -1

            if left_can_react and right_can_react:
                total_o = omega_factors[left_n] + omega_factors[right_n]
                if total_o > 1e-9:
                    chosen_n = left_n if np.random.rand() < omega_factors[left_n] / total_o else right_n
                else:
                    chosen_n = left_n if np.random.rand() < 0.5 else right_n
            elif left_can_react: chosen_n = left_n
            elif right_can_react: chosen_n = right_n

            if chosen_n != -1:
                chain[initial_reaction_idx] = REACTED
                chain[chosen_n] = REACTED
                site1, site2 = initial_reaction_idx, chosen_n
                is_available[initial_reaction_idx] = False
                is_available[chosen_n] = False
                num_available -= 2
            else: 
                chain[initial_reaction_idx] = ISOLATED
                is_available[initial_reaction_idx] = False
                num_available -= 1

            if site1 != -1:
                if site1 > site2: site1, site2 = site2, site1
                idx_l = site1 - 1
                if 0 <= idx_l < n_sites and is_available[idx_l]:
                    if (idx_l - 1 < 0 or not is_available[idx_l - 1]):
                        chain[idx_l] = ISOLATED
                        is_available[idx_l] = False
                        num_available -= 1
                    else:
                        alpha_factors[idx_l] = max(0.0, alpha_factors[idx_l] - alpha_dec)
                        omega_factors[idx_l] = max(0.0, omega_factors[idx_l] - omega_dec)

                idx_r = site2 + 1
                if 0 <= idx_r < n_sites and is_available[idx_r]:
                    if (idx_r + 1 >= n_sites or not is_available[idx_r + 1]):
                        chain[idx_r] = ISOLATED
                        is_available[idx_r] = False
                        num_available -= 1
                    else:
                        alpha_factors[idx_r] = max(0.0, alpha_factors[idx_r] - alpha_dec)
                        omega_factors[idx_r] = max(0.0, omega_factors[idx_r] - omega_dec)

    isolated_indices = np.where(chain == ISOLATED)[0]
    return chain, isolated_indices, total_collisions

@jit(nopython=True, parallel=True)
def run_parallel_simulations(num_sims, n_sites_array, inert_groups_array, act_prob, a_dec, o_dec, i_alpha, i_omega, iso_act):
    isolated_fractions = np.empty(num_sims, dtype=np.float64)
    total_collisions = np.empty(num_sims, dtype=np.int64)
    for i in prange(num_sims):
        n_sites = n_sites_array[i]
        num_inert = inert_groups_array[i]
        num_initially_available = n_sites - num_inert
        _, isolated_run, collisions_run = run_single_simulation(
            n_sites, num_inert, act_prob, a_dec, o_dec, i_alpha, i_omega, iso_act
        )
        isolated_fractions[i] = len(isolated_run) / num_initially_available if num_initially_available > 0 else 0.0
        total_collisions[i] = collisions_run
    return isolated_fractions, total_collisions

# =============================================================================
# 3. 主流程与 GPC 数据处理 (Main Execution & GPC Data Processing)
# =============================================================================

if __name__ == '__main__':
    print(f"--- Initiating {MODEL_NAME} ---")
    
    # ---------------------------------------------------------
    # 步骤 A：链长生成与抽样 (Chain Length Generation)
    # ---------------------------------------------------------
    if MW_MODE == 'GPC':
        print(f"[*] Reading Empirical GPC Data from: {GPC_FILE_PATH}")
        try:
            gpc_df = pd.read_excel(GPC_FILE_PATH)
            # 假设第一列为 logM，第二列为 dw/dlogM (兼容格式差异，直接用索引取列)
            logM_data = gpc_df.iloc[:, 0].values
            dw_dlogM_data = gpc_df.iloc[:, 1].values
            
            # 1. 物理量转换: logM -> 分子量 (M) -> 聚合度 (Degree of Polymerization, N)
            M_data = 10 ** logM_data
            chain_length_data = np.round(M_data / MONOMER_MASS).astype(np.int64)
            
            # 2. 将质量分数 (Weight Fraction) 转换为 数量分数 (Number Fraction)
            # 因为蒙特卡洛抽样是按“分子个数”抽的，长链占据的质量大但个数少
            number_fraction = dw_dlogM_data / M_data
            
            # 过滤掉可能的基线噪声产生的负值
            number_fraction = np.maximum(number_fraction, 0)
            
            # 归一化为概率分布 (Probability Distribution)
            probabilities = number_fraction / np.sum(number_fraction)
            
            # 3. 按照 GPC 的数量分数分布，随机抽样生成模拟所需的分子链集合
            raw_n_sites = np.random.choice(chain_length_data, size=NUM_SIMULATIONS, p=probabilities)
            
            # 防止出现极端的非物理短链（至少包含 2 个单体才能反应）
            n_sites_array = np.clip(raw_n_sites, 2, None)
            
            print(f"[*] GPC Sampling Complete. Mean Chain Length: {np.mean(n_sites_array):.1f}")
            
        except Exception as e:
            print(f"[!] Error reading GPC file: {e}")
            exit(1)
            
    elif MW_MODE == 'LOGNORMAL':
        print(f"[*] Theoretical Log-Normal Mode (Mean DPn={MW_MEAN}, PDI={MW_PDI})")
        sigma = np.sqrt(np.log(MW_PDI))
        mu = np.log(MW_MEAN) - (sigma**2) / 2
        raw_n_sites = np.random.lognormal(mean=mu, sigma=sigma, size=NUM_SIMULATIONS)
        n_sites_array = np.clip(np.round(raw_n_sites), 2, None).astype(np.int64)
        
    else:
        print(f"[*] Fixed Chain Length Mode (N={N_SITES_FIXED})")
        n_sites_array = np.full(NUM_SIMULATIONS, N_SITES_FIXED, dtype=np.int64)

    # ---------------------------------------------------------
    # 步骤 B：模拟执行 (Simulation Execution)
    # ---------------------------------------------------------
    # 动态分配每条链上的惰性基团（向下取整）
    inert_groups_array = np.floor(n_sites_array * INERT_FRACTION).astype(np.int64)

    print(f"Total Monte Carlo Iterations: {NUM_SIMULATIONS}")
    print("\nWarm-up: Pre-compiling Numba functions...")
    _ = run_parallel_simulations(2, n_sites_array[:2], inert_groups_array[:2], ACTIVATION_PROBABILITY, ALPHA_DECREASE, OMEGA_DECREASE, INITIAL_ALPHA, INITIAL_OMEGA, ISOLATED_ACTIVITY)
    
    start_time = time.time()
    print("Executing parallel simulations...")
    iso_fracs, coll_counts = run_parallel_simulations(
        NUM_SIMULATIONS, n_sites_array, inert_groups_array,
        ACTIVATION_PROBABILITY, ALPHA_DECREASE, OMEGA_DECREASE,
        INITIAL_ALPHA, INITIAL_OMEGA, ISOLATED_ACTIVITY
    )
    print(f"Simulations completed in: {time.time() - start_time:.2f} seconds.")

    # ---------------------------------------------------------
    # 步骤 C：数据导出与可视化 (Export & Visualization)
    # ---------------------------------------------------------
    EXCEL_FILENAME = f'sim_Results_{MW_MODE}_Sims{NUM_SIMULATIONS}.xlsx'
    df_out = pd.DataFrame({
        'Chain_Length': n_sites_array,
        'Inert_Groups': inert_groups_array,
        'Isolated_Fraction': iso_fracs,
        'Total_Collisions': coll_counts
    })
    df_out.to_excel(EXCEL_FILENAME, index=False)
    print(f"Results exported to: {EXCEL_FILENAME}")

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 论文用图 1: 验证 GPC 抽样分布的准确性
    if MW_MODE == 'GPC':
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 绘制 Monte Carlo 采样出的链长分布 (直方图)
        ax1.hist(n_sites_array, bins=100, color='skyblue', density=True, alpha=0.7, label='Sampled Chain Lengths (Simulation)')
        ax1.set_xlabel('Degree of Polymerization (Chain Length, $N$)', fontsize=12)
        ax1.set_ylabel('Probability Density (Sampled Number Fraction)', color='tab:blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # 将原始 GPC 数据按链长 N 映射上去比对 (理论的数量分布)
        ax2 = ax1.twinx()
        ax2.plot(chain_length_data, probabilities, color='red', lw=2, label='Empirical Number Fraction (GPC derived)')
        ax2.set_ylabel('Empirical Probability', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Validation of Monte Carlo GPC Sampling', fontsize=14)
        fig.tight_layout()
        plt.savefig("GPC_Sampling_Validation.png", dpi=300)
        plt.show()

    # 论文用图 2: 最终的孤立反应基团比例分布
    mean_f, std_f = np.mean(iso_fracs), np.std(iso_fracs)
    plt.figure(figsize=(10, 6))
    plt.hist(iso_fracs, bins=50, density=True, alpha=0.7, color='mediumseagreen', label='Simulation Data')
    
    # 添加正态拟合曲线
    x = np.linspace(np.min(iso_fracs), np.max(iso_fracs), 100)
    plt.plot(x, norm.pdf(x, mean_f, std_f), 'r-', lw=2, label=f'Fit (Mean={mean_f:.4f}, Std={std_f:.4f})')
    
    plt.title(f"Isolated Fraction Distribution ({MW_MODE} Mode, N={NUM_SIMULATIONS})", fontsize=14)
    plt.xlabel("Isolated Fraction", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend()
    plt.savefig(f"Isolated_Fraction_{MW_MODE}.png", dpi=300)
    plt.show()