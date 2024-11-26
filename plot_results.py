import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("pastel")

def load_results_to_dataframe(results_path:str) -> pd.DataFrame:
    """
    Load the CSV file located at results_path into a Pandas DataFrame.

    Parameters:
        results_path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    df = pd.read_csv(results_path, delimiter=';')
    return df

def get_value(df:pd.DataFrame, label:str, metric:str, statistic:str) -> float:
    """
    Extract the VALUE for a specific LABEL, METRIC, and STATISTIC from the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        label (str): The value for the LABEL column.
        metric (str): The value for the METRIC column.
        statistic (str): The value for the STATISTIC column.

    Returns:
        float: The VALUE associated with the specified LABEL, METRIC, and STATISTIC.
    """
    # Apply filtering based on LABEL, METRIC, and STATISTIC
    result = df[(df['LABEL'] == label) & (df['METRIC'] == metric) & (df['STATISTIC'] == statistic)]
    
    # If result is found, return the VALUE. Otherwise, return None or raise an exception.
    if not result.empty:
        return result.iloc[0]['VALUE']
    else:
        raise RuntimeError(f"No entry for {label} + {metric} + {statistic}.")

def main(path_to_results:str):
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot
    result_folders_names:list[str] = os.listdir(path_to_results)
    results:list[pd.DataFrame] = []

    for folder_name in result_folders_names:
        file_path:str = folder_name + '/results_summary.csv'
        path = os.path.join(path_to_results, file_path)
        results.append(load_results_to_dataframe(path))

    dice_means:list[float] = []
    for res in results:
        dice_means.append(get_value(res, 'Amygdala', 'DICE', 'MEAN'))

    sns.barplot(dice_means)
    plt.show()

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    DEFAULT_RES_PATH = 'mia-result/' # Watch out this is relative and depends on from where you execute.

    main(DEFAULT_RES_PATH)
