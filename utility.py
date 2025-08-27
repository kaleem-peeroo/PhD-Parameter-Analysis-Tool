import warnings
import signal
import plotext as pltt
import click
import time
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

from scipy import stats
from rich.pretty import pprint
from rich.markdown import Markdown
from rich import print
from rich.console import Console
console = Console()

# Mute FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def create_markdown_table(
    headers: list[str] = [""],
    rows: list[list[str]] = [[""]]
):
    """
    Convert a Rich Table to a Markdown table string.
    """
    # Build Markdown table
    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("|" + "|".join(["-" * len(header) for header in headers]) + "|")
    
    for row in rows:
        # Remove rich colours
        row = [str(cell).replace("[green]", "").replace("[/green]", "") for cell in row]
        md_lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md_lines)

def get_date_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_param_title(param):
    if param.lower() == 'datalen_bytes':
        return "Data Length (Bytes)"

    elif param.lower() == "pub_count":
        return "Publisher Count" 

    elif param.lower() == "sub_count":
        return "Subscriber Count" 

    elif param.lower() == "use_multicast":
        return "Multicast"

    elif param.lower() == "use_reliable":
        return "Reliability"

    elif param.lower() == "durability":
        return "Durability"

    else:
        console.print(f"No title found for parameter {param}", style="bold yellow")
        click.confirm("Thanks for letting me know", default=True)
        return param.capitalize()

def get_ds_df(ds_name: str = ""):
    if ds_name == "":
        console.print("No dataset found", style="bold red")
        raise ValueError("No dataset found")

    if not os.path.exists("./output"):
        console.print("No output directory found", style="bold red")
        raise ValueError("No output directory found")

    dir_contents = os.listdir("./output")
    dir_contents = [os.path.join("./output", x) for x in dir_contents]

    ds_files = [f for f in os.listdir("./output") if f.endswith("_analysed.parquet")]
    if len(ds_files) == 0:
        console.print("No dataset files found", style="bold red")
        raise ValueError("No dataset files found")

    # print(f"Getting dataset df for {ds_name}")

    if ds_name == "5PI Mcast 1P3S":
        ds_filename = "1P_3S_Multicast_Exploration_analysed.parquet"

    elif ds_name == "5PI Mcast 2P2S":
        ds_filename = "2P_2S_QoS_Capture_analysed.parquet"

    elif ds_name == "5PI QoS 1P3S":
        ds_filename = "1P_3S_QoS_Capture_analysed.parquet"

    elif ds_name == "5PI QoS 2P2S":
        ds_filename = "2P_2S_QoS_Capture_analysed.parquet"

    elif ds_name == "5PI QoS 3P1S":
        ds_filename = "3P_1S_QoS_Capture_analysed.parquet"

    elif ds_name == "3PI Mcast":
        ds_filename = "3pi_mcast_exploration_analysed.parquet"

    else:
        console.print(f"Dataset {ds_name} not found", style="bold red")
        raise ValueError("Dataset not found")

    if ds_filename not in ds_files:
        console.print(f"Dataset {ds_name} not found", style="bold red")
        raise ValueError("Dataset not found")

    # console.print(f"Dataset {ds_name} found", style="green")

    ds_df = pd.read_parquet(f"./output/{ds_filename}")
    return ds_df
    
def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    return ListedColormap(initial_cm)

def get_dataset(dataset_name, datasets):
    if datasets is None:
        console.print("No datasets found", style="bold red")
        raise ValueError("No datasets found")

    if len(datasets) == 0:
        console.print("No datasets found", style="bold red")
        raise ValueError("No datasets found")

    if not isinstance(dataset_name, str):
        console.print("Dataset name must be a string", style="bold red")
        raise ValueError("Dataset name must be a string")

    ds_names = [ds.get_name().lower() for ds in datasets]

    if dataset_name.lower() not in ds_names:
        console.print(
            f"Dataset {dataset_name} not found. Available datasets:\n\t{ds_names}",
            style="bold red"
        )
        raise ValueError("Dataset not found")

    return [ds for ds in datasets if ds.get_name().lower() == dataset_name.lower()][0]

def get_tp_col(df):
    if 'avg_mbps_per_sub' in df.columns:
        tp_col = 'avg_mbps_per_sub'

    elif 'avg_mbps' in df.columns:
        tp_col = 'avg_mbps'

    else:
        raise ValueError("No throughput column found")

    return tp_col

def filter_col_variants(df, colname):
    if colname not in df.columns:
        raise ValueError(f"Column {colname} not found")

    temp_df = df.copy()

    temp_df = temp_df[[
        'experiment_name',
        'datalen_bytes',
        'duration_secs',
        'pub_count',
        'sub_count',
        'use_reliable',
        'use_multicast',
        'durability',
        'latency_count',
    ]]

    group_cols = [
        col for col in temp_df.columns if col not in [
            'experiment_name', 
            colname
        ]
    ]

    filtered_exp_names = []
    grouped_df = temp_df.groupby(group_cols)
    for _, group in grouped_df:
        if group[colname].nunique() == 2:
            if group['experiment_name'].nunique() == 2:
                filtered_exp_names.extend(group['experiment_name'].unique())

    filtered_df = df[df['experiment_name'].isin(filtered_exp_names)]

    return filtered_df

def replace_nth(string, substring, replacement, n):
    parts = string.split(substring, n)

    if n > len(parts) - 1:
        return string

    return substring.join(parts[:n]) + replacement + substring.join(parts[n:])

def get_param_variations(temp_df, variation_param):

    param_variations = temp_df[variation_param].unique()

    # Parse values as integers for sorting purposes
    for param_variation in param_variations:
        try:
            param_variation = int(param_variation)
        except ValueError:
            pass
    param_variations = sorted(param_variations)

    if len(param_variations) < 2:
        console.print(
            f"There aren't enough {variation_param} variations to categorise.", 
            style="bold red"
        )

        console.print(
            f"Found only {len(param_variations)} variations {param_variations}", 
            style="bold red"
        )

        return None
        
    return param_variations

def get_frequency_df(df, colname):
    value_counts = df[colname].value_counts()
    proportions = df[colname].value_counts(normalize=True)

    frequency_df = pd.DataFrame({
        'value': value_counts.index,
        'count': value_counts.values,
        'proportion': proportions.values
    })

    frequency_df['percent'] = frequency_df['proportion'].apply(
        lambda x: f"{x * 100:.2f}%"
    )

    return frequency_df

def signal_handler(signum, frame):
    raise TimeoutError("Function timed out")
    
def get_metric_col(metric, df):
    if metric == "latency":
        metric_col = 'latency_us'

    else:
        metric_col = get_tp_col(df)

    if metric_col not in df.columns:
        raise ValueError(f"Column {metric_col} not found")

    return metric_col

def calculate_magnitude(
    varied_variable,
    varied_value,
    percentile,
    df,
    metric
):
    metric_col = get_metric_col(metric, df)

    if len(df) == 0:
        return None

    percentile_value = np.percentile(
        df[df[varied_variable] == varied_value][metric_col],
        percentile
    )

    return percentile_value

def get_category_col(df):
    category_cols = [col for col in df.columns if 'most_performant' in col]

    if len(category_cols) == 0:
        raise ValueError("No category columns found")

    if len(category_cols) > 1:
        raise ValueError("Multiple category columns found")

    return category_cols[0]

