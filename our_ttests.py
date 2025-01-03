from scipy.stats import ttest_rel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    fname = "test_set_results.csv"
    data = pd.read_csv(fname)

    """ttest on DICE"""
    dice_t, dice_p = ttest_rel(data['pp_test_dice'].array, data['raw_test_dice'], alternative='greater')
    print(f"T-test on DICE: stat = {dice_t} p_val = {dice_p}")

    """ttest on HSDRF"""
    hsdrf_t, hsdrf_p = ttest_rel(data['pp_test_hdrfdst'].array, data['raw_test_hdrfdst'], alternative='less')
    print(f"T-test on HSDRF: stat = {hsdrf_t} p_val = {hsdrf_p}")

    """RESULT"""
    """
    T-test on DICE: stat = 6.607596862514486 p_val = 7.076675270039633e-08
    -> Yes the dice is significantly higher

    T-test on HSDRF: stat = -6.4322614779670415 p_val = 1.1889691139870752e-07
    -> Yes the HSDRF is significantly lower
    """

    """PLOTTING FOR FUN"""

    """BOXPLOT"""
    plt.figure(figsize=(6,6))
    sns.boxplot([data['raw_test_dice'], data['pp_test_dice']], color='black', width=0.2, showfliers=False, fill=False, # White boxes with black edges
        medianprops=dict(color='black', linewidth=3), zorder=3 )
    
    sns.stripplot([data['pp_test_dice'], data['raw_test_dice']], color='gray', alpha=0.6, jitter=0.0, size=6, zorder=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Line', zorder=1)
    plt.ylabel("Dice coefficient [-]")
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    """VIOLINPLOT"""
    # sns.violinplot([data['pp_test_dice'], data['raw_test_dice']], fill=False, split=True, inner=None, dodge=False, bw=0.8,
    #             linewidth=1, scale="width", width=0.8, zorder=2)
    
    # plt.show()
