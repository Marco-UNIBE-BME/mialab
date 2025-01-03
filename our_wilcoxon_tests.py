from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    fname = "test_set_results.csv"
    data = pd.read_csv(fname)

    """wilcoxon on DICE"""
    dice_t, dice_p = wilcoxon(data['pp_test_dice'].array, data['raw_test_dice'], alternative='greater')
    print(f"T-test on DICE: stat = {dice_t} p_val = {dice_p}")

    """wilcoxon on HSDRF"""
    hsdrf_t, hsdrf_p = wilcoxon(data['pp_test_hdrfdst'].array, data['raw_test_hdrfdst'], alternative='less')
    print(f"T-test on HSDRF: stat = {hsdrf_t} p_val = {hsdrf_p}")

    """RESULT"""
    """
    Wilcoxon 'greater' on DICE: stat = 461.0 p_val = 1.2517834006034737e-06
    -> Yes the dice is significantly higher


    Wilcoxon 'less' on HSDRF: stat = 58.0 p_val = 3.503204386142291e-05
    -> Yes the HSDRF is significantly lower
    """

    """PLOTTING FOR FUN"""

    """BOXPLOT"""
    # plt.figure(figsize=(6,6))
    # plt.title("Results on the Test-Set (n=7)", fontsize=20)
    # sns.boxplot([data['raw_test_dice'], data['pp_test_dice']], color='black', width=0.4, showfliers=False, fill=False, # White boxes with black edges
    #     medianprops=dict(color='orange', linewidth=3), zorder=3 )
    # plt.axhline(np.median(data['raw_test_dice']), color='black', linestyle='--', linewidth=1.5, label='raw median', zorder=1, alpha=0.5)
    # sns.stripplot([data['pp_test_dice'], data['raw_test_dice']], color='gray', alpha=0.6, jitter=0.0, size=6, zorder=2)
    # plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Line', zorder=1)
    # plt.ylabel("Dice coefficient [-]", fontsize=18)
    # plt.xlabel("Approach", fontsize=18)
    # plt.ylim([0, 1])
    # plt.yticks(fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(6,6))
    plt.title("Results on the Test-Set (n=7*5)", fontsize=20)

    # sns.boxplot([data['raw_test_dice'], data['pp_test_dice']], color='black', width=0.4, showfliers=False, fill=False, # White boxes with black edges
    #     medianprops=dict(color='orange', linewidth=3), zorder=3 )
    # plt.axhline(np.median(data['raw_test_dice']), color='black', linestyle='--', linewidth=1.5, label='raw median', zorder=1, alpha=0.5)
    # sns.stripplot([data['raw_test_dice'], data['pp_test_dice']], color='gray', alpha=0.3, jitter=0.0, size=6, zorder=2)

    sns.boxplot([data['raw_test_hdrfdst'], data['pp_test_hdrfdst']], color='black', width=0.4, showfliers=False, fill=False, # White boxes with black edges
        medianprops=dict(color='orange', linewidth=3), zorder=3 )
    plt.axhline(np.median(data['raw_test_hdrfdst']), color='black', linestyle='--', linewidth=1.5, label='raw median', zorder=1, alpha=0.5)
    sns.stripplot([data['raw_test_hdrfdst'], data['pp_test_hdrfdst']], color='gray', alpha=0.3, jitter=0.0, size=6, zorder=2)

    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Line', zorder=1)
    # plt.ylabel("Dice coefficient [-]", fontsize=18)
    plt.ylabel("Hausdorff Dist. [mm]", fontsize=18)
    plt.xlabel("Approach", fontsize=18)
    # plt.ylim([0, 1])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # """VIOLINPLOT"""
    # sns.violinplot([data['pp_test_dice'], data['raw_test_dice']], fill=False, split=True, inner=None, dodge=False, bw=0.8,
    #             linewidth=1, scale="width", width=0.8, zorder=2)
    
    plt.show()
