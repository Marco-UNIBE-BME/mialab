import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

fname = "fillhole_res"

print(os.listdir(fname))

files = os.listdir(fname)
dice_dict = {}
hsdrf_dict = {}

for file in files:
    if '_50' in file:
        key = '2d50'
    elif '_100' in file:
        key = '2d100'
    elif '_150' in file:
        key = '2d150'
    elif '_200' in file:
        key = '2d200'
    elif '_250' in file:
        key = '2d250'
    elif '_bin' in file:
        key = 'bin_geo'
    else:
        raise RuntimeError
    
    with open(fname+'/'+file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        header = next(reader)
        rows = [row for row in reader]
    df = pd.DataFrame(rows, columns=header)
    filtered_values = df[(df['METRIC'] == 'DICE') & (df['STATISTIC'] == 'MEAN')]['VALUE']
    dice_dict[key] = filtered_values.astype(float)

    filtered_values = df[(df['METRIC'] == 'HDRFDST') & (df['STATISTIC'] == 'MEAN')]['VALUE']
    hsdrf_dict[key] = filtered_values.astype(float)
print(dice_dict['2d50'],  len(dice_dict['2d50']))

data = pd.DataFrame({
    'Mean Dice': dice_dict['2d50'] + dice_dict['2d100'] + dice_dict['2d150'] + dice_dict['2d200'] + dice_dict['2d250'] + dice_dict['bin_geo'],
    'Method': ['2d50'] * len(dice_dict['2d50']) + ['2d100'] * len(dice_dict['2d50']) + ['2d150'] * len(dice_dict['2d50']) + ['2d200'] * len(dice_dict['2d50']) + ['2d250'] * len(dice_dict['2d50']) + ['bin_geo'] * len(dice_dict['2d50']),
})

sns.boxplot([dice_dict['2d50'], dice_dict['2d100']], dodge=True)
plt.show()
# data = []
# for key, values in dice_dict.items():
#     print(key)
#     for value in values:
#         data.append({'Key': key, 'Value': float(value)})  # Ensure values are numeric

# df = pd.DataFrame(data)

# # Plot the boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x='Key', y='Value', palette="muted")
# plt.title("Boxplot of Dice Metrics", fontsize=16, fontweight='bold', pad=20)
# plt.xlabel("Keys", fontsize=14)
# plt.ylabel("Dice Values", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.tight_layout()
# plt.show()
