"""
Tool for analysing which parameter (or combinations of parameters) values are more
performant.
"""

import os
import sys
import ast
import logging
import pandas as pd
import numpy as np
import timer
import warnings
import click
import matplotlib.pyplot as plt
import plotext as pltx
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from utility import get_date_timestamp, create_markdown_table
from config import d_config
from collections import Counter
from pync import Notifier
from math import sqrt
from scipy import stats as scistats
from scipy.stats import chi2_contingency
from itertools import product, combinations
from rich import print
from rich.pretty import pprint
from rich.markdown import Markdown
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler

console = Console()
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=False,
        )
    ],
)

lg = logging.getLogger("rich")
plt_logger = logging.getLogger("matplotlib")
plt_logger.setLevel(logging.ERROR)

PARAM_COLS = [
    "datalen_bytes",
    "pub_count",
    "sub_count",
    "use_reliable",
    "use_multicast",
    "durability",
]


def main(D_CONFIG):
    df = pd.read_parquet(d_config["data_path"])
    lg.debug(f"Loaded dataset ({len(df.index):,}) rows)")

    if not validate_experiment_names(df):
        update_experiment_names(df)

    df["datalen_bytes"] = df["datalen_bytes"].astype(int)
    df["pub_count"] = df["pub_count"].astype(int)
    df["sub_count"] = df["sub_count"].astype(int)
    df["durability"] = df["durability"].astype(int)

    # df = filter_datasets(df)

    # df = filter_parameter_values(df)

    # df = filter_parameter_values_interactively(df)

    if not d_config["dev_mode"]:
        varied_param_list = get_params_to_vary(df)
        if not varied_param_list:
            raise ValueError("No varied parameters found.")
        if len(varied_param_list) == 0:
            raise ValueError("No varied parameters found.")
    else:
        varied_param_list = d_config["dev_config"]["varied_param_list"]

    [
        make_needed_folders(" vs ".join(varied_param_list), metric)
        for metric in d_config["metric_columns"]
    ]

    plot_input_distribution(df)

    exp_variation_dict_list = get_variations(df, varied_param_list)
    if len(exp_variation_dict_list) == 0:
        raise ValueError("No experiment variations found.")

    variations_df = get_variations_df(df, exp_variation_dict_list, varied_param_list)
    if variations_df is None or variations_df.empty:
        raise ValueError("Variations dataframe is None or empty.")

    # parameter value combinations
    param_val_comb_dict_list = get_param_values(variations_df, varied_param_list)
    if param_val_comb_dict_list is None or len(param_val_comb_dict_list) == 0:
        raise ValueError("Parameter value combinations list is None or empty.")

    param_val_comb_dict_list = sort_list_of_dicts(param_val_comb_dict_list)
    if param_val_comb_dict_list is None or len(param_val_comb_dict_list) == 0:
        raise ValueError("Sorted parameter value combinations list is None or empty.")

    all_pairings_df = process_all_pairings(
        variations_df,
        exp_variation_dict_list,
        varied_param_list,
        d_config["metric_columns"],
        d_config["sections"],
    )

    most_perf_pairings_df = classify_pairings(
        all_pairings_df, varied_param_list, d_config
    )

    for metric in d_config["metric_columns"]:
        create_results_report(
            metric, most_perf_pairings_df, varied_param_list, d_config
        )

    if not d_config["dev_mode"]:
        if click.confirm("Do you want to explore the classifications?", default=True):
            explore_classifications(most_perf_pairings_df, varied_param_list, d_config)

    else:
        if d_config["dev_config"]["explore_classifications"]:
            explore_classifications(most_perf_pairings_df, varied_param_list, d_config)


def update_experiment_names(df):
    lg.info("Updating experiment names...")

    replacements = {
        "P_": "PUB_",
        "S_": "SUB_",
        "s_": "SEC_",
        "rel_": "REL_",
        "be_": "BE_",
        "uc_": "UC_",
        "mc_": "MC_",
        "dur_": "DUR_",
    }

    for loop_i, (original, replacement) in enumerate(replacements.items()):
        lg.info(
            "[{}/{}] Replacing {} with {}...".format(
                loop_i + 1, len(replacements), original, replacement
            )
        )

        df["experiment_name"] = df["experiment_name"].str.replace(original, replacement)

    df.to_parquet("./output/2025_02_21_datasets.parquet")


def validate_experiment_names(df):
    lg.info("Validating experiment names...")
    exp_names = df["experiment_name"].unique()

    replacements = {
        "P_": "PUB_",
        "S_": "SUB_",
        "s_": "SEC_",
        "rel_": "REL_",
        "be_": "BE_",
        "uc_": "UC_",
        "mc_": "MC_",
        "dur_": "DUR_",
    }

    for exp_name in exp_names:
        for original, replacement in replacements.items():
            if original in exp_name:
                lg.error(f"Found {original} in {exp_name}...")
                return False

    return True


def create_test_sample_df():

    if not os.path.exists("./output/datasets_testing_sample.parquet"):
        df = pd.read_parquet(DS_PATH)
        lg.info("Creating test sample dataframe...")
        df = df.sample(frac=0.01)
        df.to_parquet("./output/datasets_testing_sample.parquet")

    else:
        df = pd.read_parquet("./output/datasets_testing_sample.parquet")

    return df


def print_dict_as_table(data):
    # Initialize rich console
    console = Console()

    # Create a table
    table = Table(show_lines=True)

    # Add columns based on dictionary keys
    keys = list(data.keys())
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Add rows from the dictionary
    for i, (key, value) in enumerate(data.items()):
        if i >= 3:
            i_str = f"{i -2}. "
        else:
            i_str = ""

        table.add_row(f"{i_str}{str(key)}", str(value))

    # Print the table to the console
    console.print(table)


def describe_df(df):
    """
    What could go wrong?
    - df is empty
    - df is none
    - df is not a dataframe
    - df doesn't have 'dataset' column
    - dataset column is empty
    """
    if df is None:
        raise ValueError("The dataframe is None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a pandas dataframe.")

    if df.empty:
        raise ValueError("The dataframe is empty.")

    if "dataset" not in df.columns:
        raise ValueError("The dataframe does not have a 'dataset' column.")

    if df["dataset"].empty:
        raise ValueError("The 'dataset' column is empty.")

    print(Markdown("# Dataset Description"))

    stats = {}

    datasets = df["dataset"].unique().tolist()
    stats[f"datasets ({len(datasets)})"] = datasets

    stats["params"] = PARAM_COLS
    stats["metrics"] = METRIC_COLS

    param_stats = describe_params(df)

    stats.update(param_stats)

    return stats


def filter_datasets(df):
    lg.info(f"Filtering out {len(UNDESIRED_DATASETS)} datasets...")

    ls_ds_before = df["dataset"].unique().tolist()
    pprint(ls_ds_before)
    df = df[~df["dataset"].isin(UNDESIRED_DATASETS)]
    df = df.reset_index(drop=True)
    ls_ds_after = df["dataset"].unique().tolist()
    pprint(ls_ds_after)
    asdf

    ls_removed_datasets = list(set(ls_ds_before) - set(ls_ds_after))
    lg.info(f"Removed {len(ls_removed_datasets)} datasets: {ls_removed_datasets}")

    return df


def describe_params(df):
    # log.info("Describing parameters...")
    param_stats = {}

    for param in PARAM_COLS:
        values = df[param].unique().tolist()
        values = sorted(values)
        param_stats[f"{param} ({len(values)})"] = values

    return param_stats


def plot_param_distributions(df):
    lg.info("Plotting parameter distributions...")

    pltx.clf()
    pltx.subplots(2, 3)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # Set fontsize to 14 for plt
    plt.rcParams.update({"font.size": 14})

    for param_i, param in enumerate(PARAM_COLS):
        # Terminal Plotting with Plotext
        subplot_x = param_i % 3 + 1
        subplot_y = param_i // 3 + 1

        pltx.subplot(subplot_y, subplot_x)
        pltx.plotsize(pltx.tw() // 3, pltx.th() * 0.2)

        param_df = df[[param]].copy()
        param_df = param_df[param].astype(int)
        param_df = param_df.drop_duplicates().reset_index(drop=True)
        param_count = param_df.nunique()

        pltx.hist(param_df, bins=param_count, label=param)
        pltx.title(f"{param}")

    pltx.show()

    for param_i, param in enumerate(PARAM_COLS):
        # Matplotlib Plotting with Pyplot
        subplot_x = param_i % 3 + 1
        subplot_y = param_i // 3 + 1

        param_df = df[[param]].copy()
        param_df = param_df[param].astype(float)
        param_count = param_df.nunique()

        ax = axs[subplot_y - 1, subplot_x - 1]
        ax.hist(param_df, bins=param_count, color="#c1e7ff", rwidth=0.85)
        ax.set_title(param)

        # if param_count > 2:
        ax.set_yscale("log")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set the color of the axis, ticks, and labels to #444
        desired_color = "#999"
        ax.spines["left"].set_color(desired_color)
        ax.spines["bottom"].set_color(desired_color)
        ax.xaxis.label.set_color(desired_color)
        ax.yaxis.label.set_color(desired_color)
        ax.tick_params(axis="x", colors=desired_color)
        ax.tick_params(axis="y", colors=desired_color)

    plt.tight_layout()
    plt.savefig("./output/parameter_analysis_tool/parameter_distributions.png")
    plt.savefig("./output/parameter_analysis_tool/parameter_distributions.pdf")

    # log.info(f"Parameter distributions saved to\n\t{
    #     os.path.abspath('./output/parameter_analysis_tool/parameter_distributions.png')
    # }")


def get_variations(df, params_to_vary):
    # print(f"Getting variations for {params_to_vary}...")

    """
    1. Get all other params (unvaried params)
    2. Get all unique combinations of unvaried params
    3. For each unvaried param combination
        3.1. Get all unique values of the varied param
        3.2. If the number of unique values is less than 2, skip
        3.3. Else, add the combination to the list of variations
    4. Return the list of variations which is a df with unvaried param combinations
    """

    param_df = df[PARAM_COLS].drop_duplicates().reset_index(drop=True)
    unvaried_params = [param for param in PARAM_COLS if param not in params_to_vary]

    if not unvaried_params:
        return []

    # Group by the unvaried parameters and count the number of variations for each group.
    # A "variation" is a unique combination of the "params_to_vary".
    # The size of each group gives us this count.
    variation_counts = param_df.groupby(unvaried_params).size()

    # We are only interested in the groups that have at least 2 variations to compare.
    sufficient_variations = variation_counts[variation_counts >= 2]

    # The index of the resulting series contains the unvaried parameter combinations.
    # We can reset the index to get a DataFrame and then convert it to a list of dicts.
    variations_df = sufficient_variations.reset_index()[unvaried_params]

    return variations_df.to_dict("records")


def filter_parameter_values_interactively(df: pd.DataFrame = pd.DataFrame()):
    """
    What could go wrong?
    - df is none
    - df is empty
    - df is not a dataframe
    """

    if df is None:
        raise ValueError("The dataframe is None.")

    if df.empty:
        raise ValueError("The dataframe is empty.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a pandas dataframe.")

    # TODO: Uncomment if you ever need to plot the distributions in terminal
    # plot_param_distributions(df)
    stats = describe_df(df)
    print_dict_as_table(stats)

    click.secho("0. Go back", fg="red", bold=True)

    for param_col_i, param_col in enumerate(PARAM_COLS):
        click.echo(f"{param_col_i + 1}. {param_col}")

    click.secho(f"{len(PARAM_COLS) + 1}. Continue", fg="green", bold=True)

    if not d_config["dev_mode"]:
        user_choice = click.prompt(
            "Which parameter do you want to {}?".format(
                click.style(
                    "filter",
                    fg="black",
                    bg="green",
                    bold=True,
                )
            ),
            type=click.IntRange(0, len(PARAM_COLS) + 1),
            default=len(PARAM_COLS) + 1,
        )

    else:
        user_choice = d_config["dev_config"]["filter_option"]

    if user_choice == 0:
        return df

    elif user_choice == len(PARAM_COLS) + 1:
        return df

    else:
        df = filter_parameter_value(df, PARAM_COLS[user_choice - 1])

    while True:
        # TODO: Uncomment if you ever need to plot the distributions in terminal
        # plot_param_distributions(df)
        stats = describe_df(df)
        print_dict_as_table(stats)

        click.secho("0. Go back", fg="red", bold=True)

        for param_col_i, param_col in enumerate(PARAM_COLS):
            click.echo(f"{param_col_i + 1}. {param_col}")

        click.secho(
            f"{len(d_config['parameter_columns']) + 1}. Continue", fg="green", bold=True
        )

        user_choice = click.prompt(
            "Which parameter do you want to {}?".format(
                click.style(
                    "filter",
                    fg="black",
                    bg="green",
                    bold=True,
                )
            ),
            type=click.IntRange(0, len(PARAM_COLS) + 1),
            default=len(PARAM_COLS) + 1,
        )

        if user_choice == 0:
            return df

        elif user_choice == len(PARAM_COLS) + 1:
            return df

        else:
            df = filter_parameter_value(df, PARAM_COLS[user_choice - 1])


def get_filter_type(df: pd.DataFrame = pd.DataFrame(), param: str = ""):
    if df.empty:
        raise ValueError("The dataframe is empty.")

    if param == "":
        raise ValueError("The parameter is empty.")

    if isinstance(df[param].iloc[0], np.int64):
        filter_types = ["<=", ">=", "!=", "=="]

    elif isinstance(df[param].iloc[0], str):
        filter_types = ["!=", "=="]

    elif isinstance(df[param].iloc[0], np.bool):
        filter_types = ["!=", "=="]

    elif isinstance(df[param].iloc[0], float):
        filter_types = ["<=", ">=", "!=", "=="]

    else:
        raise ValueError(
            f"{param} is a {type(df[param].iloc[0])} which is not supported."
        )

    filter_types.append("Custom List")

    click.secho("0. Go back", fg="red", bold=True)
    for filter_i, filter_type in enumerate(filter_types):
        click.echo(f"{filter_i + 1}. {filter_type}")
    click.secho("6. Continue", fg="green", bold=True)

    user_choice = click.prompt("How would you like to filter the parameter?", type=int)

    filter_type = ""

    if user_choice < 0 or user_choice > 6:
        raise ValueError("Invalid choice.")

    if user_choice == 0:
        return ""

    elif user_choice == 1:
        filter_type = "<="

    elif user_choice == 2:
        filter_type = ">="

    elif user_choice == 3:
        filter_type = "!="

    elif user_choice == 4:
        filter_type = "=="

    elif user_choice == 5:
        filter_type = "Custom List"

    else:
        return ""

    return filter_type


def print_param_values_table(
    df: pd.DataFrame = pd.DataFrame(),
    param: str = "",
    filter_type: str = "",
    filter_criteria: str = "",
):
    if df.empty:
        raise ValueError("The dataframe is empty.")

    if param == "":
        raise ValueError("The parameter is empty.")

    if param not in PARAM_COLS:
        raise ValueError("The parameter is not in the dataframe.")

    values = sorted(df[param].unique().tolist())
    filtered_values = values

    if filter_criteria and filter_type:
        if filter_type == "Custom List":
            filter_criteria_list = filter_criteria.split(",")
            filter_criteria_list = [value.strip() for value in filter_criteria_list]
            filtered_values = [
                value for value in values if str(value) in filter_criteria_list
            ]

            if isinstance(df[param].iloc[0], np.int64):
                filtered_values = [int(value) for value in filtered_values]

            elif isinstance(df[param].iloc[0], np.bool):
                filtered_values = [bool(value) for value in filtered_values]

            else:
                filtered_values = [str(value) for value in filtered_values]

        else:
            if isinstance(df[param].iloc[0], np.int64) or isinstance(
                df[param].iloc[0], float
            ):
                values = [int(value) for value in values]
                filter_criteria = float(filter_criteria)

                if filter_type == "<=":
                    filtered_values = [
                        value for value in values if value <= filter_criteria
                    ]

                elif filter_type == ">=":
                    filtered_values = [
                        value for value in values if value >= filter_criteria
                    ]

                elif filter_type == "!=":
                    filtered_values = [
                        value for value in values if value != filter_criteria
                    ]

                elif filter_type == "==":
                    filtered_values = [
                        value for value in values if value == filter_criteria
                    ]

            elif isinstance(df[param].iloc[0], str):
                values = [str(value) for value in values]
                filter_criteria = str(filter_criteria)

                if filter_type == "!=":
                    filtered_values = [
                        value for value in values if value != filter_criteria
                    ]

                elif filter_type == "==":
                    filtered_values = [
                        value for value in values if value == filter_criteria
                    ]

            elif isinstance(df[param].iloc[0], np.bool):
                values = [bool(value) for value in values]
                filter_criteria = bool(filter_criteria)

                if filter_type == "!=":
                    filtered_values = [
                        value for value in values if value != filter_criteria
                    ]

                elif filter_type == "==":
                    filtered_values = [
                        value for value in values if value == filter_criteria
                    ]

            else:
                raise ValueError(
                    f"{param} is a {type(df[param].iloc[0])} which is not supported."
                )

    colored_values = []
    for value in values:
        if value in filtered_values:
            colored_values.append(f"[bold green]{value}[/bold green]")
        else:
            colored_values.append(f"[bold red]{value}[/bold red]")

    filtered_val_str = ", ".join(colored_values)

    param_val_table = Table(show_header=False)
    param_val_table.add_row(param, filtered_val_str)
    console.print(param_val_table)


def get_params_to_vary(df):
    params_to_vary = []

    click.secho("0. Exit", fg="red", bold=True)
    for param_i, param in enumerate(PARAM_COLS):
        click.echo(f"{param_i + 1}. {param}")

    click.secho(
        f"{len(d_config['parameter_columns']) + 1}. Continue", fg="green", bold=True
    )

    user_param_i = click.prompt(
        "Which parameter do you want to {}?".format(
            click.style(
                "vary",
                fg="black",
                bg="green",
                bold=True,
            )
        ),
        type=click.IntRange(0, len(PARAM_COLS) + 1),
        default=len(PARAM_COLS) + 1,
    )

    if user_param_i == 0 or user_param_i == len(PARAM_COLS) + 1:
        return []

    params_to_vary.append(PARAM_COLS[user_param_i - 1])
    variations = get_variations(df, params_to_vary)
    console.print(f"{params_to_vary}: {len(variations)} variations", style="bold green")

    while True:
        click.secho("0) Stop", fg="red", bold=True)
        for param_i, param in enumerate(PARAM_COLS):
            click.echo(f"{param_i + 1}. {param}")

        click.secho(
            f"{len(d_config['parameter_columns']) + 1}. Continue", fg="green", bold=True
        )
        click.secho(
            f"{len(d_config['parameter_columns']) + 2}. View Variations",
            fg="green",
            bold=True,
        )

        user_param_i = click.prompt(
            "Which parameter do you want to {}?".format(
                click.style(
                    "vary",
                    fg="black",
                    bg="green",
                    bold=True,
                )
            ),
            type=click.IntRange(0, len(PARAM_COLS) + 2),
            default=len(PARAM_COLS) + 1,
        )

        if user_param_i == 0 or user_param_i == len(PARAM_COLS) + 1:
            break

        elif user_param_i > 0 and user_param_i < len(PARAM_COLS) + 1:
            params_to_vary.append(PARAM_COLS[user_param_i - 1])

        variations = get_variations(df, params_to_vary)

        if len(variations) == 0:
            console.print(f"{params_to_vary}: No variations found", style="bold red")

        else:
            console.print(
                f"{params_to_vary}: {len(variations)} variations", style="bold green"
            )

            if user_param_i == len(PARAM_COLS) + 2:
                pprint(variations)

    return params_to_vary


def check_for_variation_file(params_to_vary):
    lg.info(f"Checking for previous variation files for {params_to_vary}...")

    param_str = " vs ".join(params_to_vary)
    variation_dir = f"./output/parameter_analysis_tool/{param_str}/variations/"

    if not os.path.exists(variation_dir):
        return None

    variation_files = os.listdir(variation_dir)
    variation_files = [_ for _ in variation_files if _.endswith("variations.parquet")]
    if len(variation_files) == 0:
        return None

    variation_files = sorted(variation_files, reverse=True)
    return os.path.join(variation_dir, variation_files[0])


def get_variations_df(df, variations, params_to_vary):
    """
    What could go wrong?
    - df is none
    - df is empty
    - variations is none
    - variations is empty
    - params_to_vary is none
    - params_to_vary is empty
    - variations is not a list
    - variations is not a list of dicts
    - params_to_vary is not a list
    - params_to_vary is not a list of strings
    - params_to_vary contains invalid params
    - params_to_vary contains params that are not in df
    """

    existing_variation_file = check_for_variation_file(params_to_vary)
    if not d_config["overwrite_processing"]["variations"] and existing_variation_file:
        lg.info("Variation file exists. Loading...")
        return pd.read_parquet(existing_variation_file)

    if not variations:
        lg.warning(
            "No variations provided to get_variations_df. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=df.columns)

    # Convert the list of variation dictionaries to a DataFrame.
    variations_df = pd.DataFrame(variations)
    unvaried_params = list(variations[0].keys())

    # Use an inner merge to filter `df`. This keeps only the rows in `df`
    # where the values in `unvaried_params` match a row in `variations_df`.
    # This is much more efficient than looping and concatenating.
    df_to_return = pd.merge(df, variations_df, on=unvaried_params, how="inner")

    param_str = " vs ".join(params_to_vary)
    filedir = "./output/parameter_analysis_tool/{}/variations/".format(param_str)
    filename = "{}_variations.parquet".format(get_date_timestamp())
    filepath = os.path.join(filedir, filename)

    write_df(df_to_return, filepath)

    lg.info(f"Variations saved to\n\t{os.path.abspath(filepath)}")

    return df_to_return


def filter_parameter_value(df: pd.DataFrame = pd.DataFrame(), param: str = ""):
    if df.empty:
        raise ValueError("The dataframe is empty.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a pandas dataframe.")

    if param == "":
        raise ValueError("The parameter is empty.")

    if param not in PARAM_COLS:
        raise ValueError("The parameter is not in the dataframe.")

    print_param_values_table(df, param)
    while True:
        filter_type = get_filter_type(df, param)
        if filter_type == "":
            return df

        filter_criteria = click.prompt("Enter the value to filter by", type=str)

        print_param_values_table(df, param, filter_type, filter_criteria)

        click.secho("0. Go back", fg="red", bold=True)
        click.secho("1. Looks good - filter it!", fg="green", bold=True)
        user_choice = click.prompt("What would you like to do?", type=int, default=1)

        if user_choice == 0:
            return df

        elif user_choice == 1:
            if filter_type == "Custom List":
                filter_criteria_list = filter_criteria.split(",")
                filter_criteria_list = [value.strip() for value in filter_criteria_list]
                filter_criteria_list = [
                    value for value in filter_criteria_list if value
                ]

                if isinstance(df[param].iloc[0], np.int64):
                    filter_criteria_list = [
                        int(value) for value in filter_criteria_list
                    ]

                elif isinstance(df[param].iloc[0], np.bool):
                    filter_criteria_list = [
                        bool(value) for value in filter_criteria_list
                    ]

                else:
                    filter_criteria_list = [
                        str(value) for value in filter_criteria_list
                    ]

                df = df[df[param].isin(filter_criteria_list)]

            else:

                if isinstance(df[param].iloc[0], np.int64) or isinstance(
                    df[param].iloc[0], float
                ):
                    filter_criteria = float(filter_criteria)

                    if filter_type == "<=":
                        df = df[df[param] <= filter_criteria]

                    elif filter_type == ">=":
                        df = df[df[param] >= filter_criteria]

                    elif filter_type == "!=":
                        df = df[df[param] != filter_criteria]

                    elif filter_type == "==":
                        df = df[df[param] == filter_criteria]

                elif isinstance(df[param].iloc[0], str):
                    filter_criteria = str(filter_criteria)

                    if filter_type == "!=":
                        df = df[df[param] != filter_criteria]

                    elif filter_type == "==":
                        df = df[df[param] == filter_criteria]

                elif isinstance(df[param].iloc[0], np.bool):
                    filter_criteria = bool(filter_criteria)

                    if filter_type == "!=":
                        df = df[df[param] != filter_criteria]

                    elif filter_type == "==":
                        df = df[df[param] == filter_criteria]

                else:
                    raise ValueError(
                        f"{param} is a {type(df[param].iloc[0])} which is not supported."
                    )

            return df

        else:
            raise ValueError("Invalid choice.")


def filter_parameter_values(df):
    lg.info("Filtering parameter values...")

    n_before = len(df)
    df = df[
        (df["datalen_bytes"].isin([100, 512, 1000, 32, 64, 128, 256, 1024]))
        # (df['pub_count'] == 1) &
        # (df['sub_count'] <= 15)
        # (df['use_reliable'] == 0) &
        # (df['use_multicast'] == 0) &
        # (df['durability'] == 0)
    ]
    n_after = len(df)

    lg.info(f"Filtered out {n_after - n_after:,.0f} values")

    return df


def write_df(df, filepath):
    """
    What could go wrong?
    - df is none
    - df is empty
    - filepath is invalid
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if filepath.endswith(".parquet"):
        df.to_parquet(filepath)

    elif filepath.endswith(".csv"):
        df.to_csv(filepath, index=False)

    else:
        raise ValueError("Invalid file extension.")


def get_param_values(df, params_to_vary):
    lg.info(f"Getting parameter values for {params_to_vary}...")

    param_values = {}

    for param in params_to_vary:
        values = df[param].unique().tolist()
        values = sorted(values)
        param_values[param] = values

    combinations = list(product(*[param_values[param] for param in params_to_vary]))

    # Add the param names back to the combinations
    updated_combinations = []
    for combination in combinations:
        updated_combinations.append(dict(zip(params_to_vary, combination)))

    return updated_combinations


def sort_list_of_dicts(list_of_dicts):
    if not list_of_dicts:
        return []

    if not isinstance(list_of_dicts, list):
        raise TypeError(f"Input is not a list:\n\t{list_of_dicts}")

    return sorted(list_of_dicts, key=lambda x: ([x[key] for key in x.keys()]))


def get_most_perf_param_dict(pairings_df, metric):
    """
    Return the pairing that is most seen in the left-most or right-most position
    """

    winner_dict_list = []
    loser_dict_list = []
    for col in ["value_a", "value_b"]:
        pairings_df[col] = pairings_df[col].astype(str)

    unique_pairings_df = pairings_df.copy()
    unique_pairings_df = unique_pairings_df.drop_duplicates(
        subset=["value_a", "value_b"]
    ).reset_index(drop=True)

    for unique_pairing_i, unique_pairing_df_row in unique_pairings_df.iterrows():
        pairing_df = pairings_df[
            (pairings_df["value_a"] == unique_pairing_df_row["value_a"])
            & (pairings_df["value_b"] == unique_pairing_df_row["value_b"])
        ]
        pairing_df = pairing_df.reset_index(drop=True)
        if len(pairing_df) == 0:
            lg.error(
                "No pairings found for {} vs {}.".format(
                    format_dict(unique_pairing_df_row["value_a"]),
                    format_dict(unique_pairing_df_row["value_b"]),
                )
            )
            continue

        all_a_on_left = all(pairing_df["a_on_left"])
        all_b_on_left = all(pairing_df["b_on_left"])

        if all_b_on_left & all_a_on_left:
            lg.error(len(pairing_df))

            lg.error(f"all_a_on_left: {all(pairing_df['a_on_left'])}")
            lg.error(f"all_a_on_left: {pairing_df['a_on_left'].value_counts()}")

            lg.error(f"all_b_on_left: {all(pairing_df['b_on_left'])}")
            lg.error(f"all_b_on_left: {pairing_df['b_on_left'].value_counts()}")

            raise ValueError("Both a and b cannot be on the left.")

        if all_a_on_left and not all_b_on_left:
            if metric == "latency_us":
                winner = unique_pairing_df_row["value_a"]
                loser_list = [unique_pairing_df_row["value_b"]]
            else:
                winner = unique_pairing_df_row["value_b"]
                loser_list = [unique_pairing_df_row["value_a"]]

        elif not all_a_on_left and all_b_on_left:
            if metric == "latency_us":
                winner = unique_pairing_df_row["value_b"]
                loser_list = [unique_pairing_df_row["value_a"]]
            else:
                winner = unique_pairing_df_row["value_a"]
                loser_list = [unique_pairing_df_row["value_b"]]

        else:  # They are intersecting
            winner = None
            loser_list = [
                unique_pairing_df_row["value_a"],
                unique_pairing_df_row["value_b"],
            ]

        if winner:
            winner_dict_list.append(str(winner))

        if loser_list:
            for loser in loser_list:
                loser_dict_list.append(str(loser))

    if not winner_dict_list:
        return None

    winner_dict_list = sorted(winner_dict_list)
    loser_dict_list = list(set(loser_dict_list))
    loser_dict_list = sorted(loser_dict_list)

    # Get the most common winner but if there are multiple, there is no winner

    # Count the number of times each winner appears
    winner_counts = Counter(winner_dict_list)

    # Get the two most common winners to check if there is a tie
    top_two_most_common = winner_counts.most_common(2)

    # If the two most common winners have the same count, there is a tie
    if (
        len(top_two_most_common) > 1
        and top_two_most_common[0][1] == top_two_most_common[1][1]
    ):
        return None

    most_performant_pairing_dict = top_two_most_common[0][0]

    # If the winner of all has intersected with another, then it hasn't won all
    if str(most_performant_pairing_dict) in loser_dict_list:
        return None

    return most_performant_pairing_dict


def process_pairings(
    param_val_comb_dict_list: list = [],
    metric: str = "latency_us",
    metric_df: pd.DataFrame = pd.DataFrame(),
):
    """
    Does the following:
        - For each pair of parameter value combinations
        - Process the pairing to get the CDFs with confidence intervals and
          intersection checks
        - Calculate the KS statistic and p-value
        - Return the processed pairings dataframe

    Args:
        - df_param_val_combs (list): List of dictionaries of parameter value combinations
        - metric (str): Metric to compare
        - metric_df (pd.DataFrame): DataFrame containing the metric values

    Returns:
        - pairings_df (pd.DataFrame): DataFrame containing the processed pairings
    """

    if not param_val_comb_dict_list:
        raise ValueError("df_param_val_combs is None or empty")

    if not metric:
        raise ValueError("metric is None or empty")

    if metric_df.empty:
        raise ValueError("metric_df is None or empty")

    pairings_list = []
    for pair_a_dict, pair_b_dict in combinations(param_val_comb_dict_list, 2):
        a_b_df, metric_a_df, metric_b_df = process_pairing_df(
            metric, pair_a_dict, pair_b_dict, metric_df
        )

        if a_b_df is None:
            continue

        ks_stat, ks_pval = scistats.ks_2samp(metric_a_df, metric_b_df)

        a_b_df["ks_stat"] = ks_stat
        a_b_df["ks_pval"] = ks_pval

        a_b_df["a"] = format_dict(pair_a_dict)
        a_b_df["b"] = format_dict(pair_b_dict)

        a_b_df["value_a"] = [pair_a_dict] * len(a_b_df)
        a_b_df["value_b"] = [pair_b_dict] * len(a_b_df)

        pairings_list.append(a_b_df)

    if not pairings_list:
        return pd.DataFrame()

    pairings_df = pd.concat(pairings_list, ignore_index=True)
    pairings_df["possible_values"] = str(param_val_comb_dict_list)
    pairings_df["metric"] = metric

    return pairings_df


def color_legend_text(
    legend,
    text_value,
    color,
):
    colored = False
    for text in legend.get_texts():
        if text.get_text() == text_value:
            text.set_color(color)
            colored = True

    if not colored:
        lg.warning(f"Could not find {text_value} in legend.")


def show_working_out(
    metric: str = "latency_us",
    metric_df: pd.DataFrame = pd.DataFrame(),
    pairings_df: pd.DataFrame = pd.DataFrame(),
    param_variations_dict_list: list = [],
    sections: list = [],
    exp_variation_dict: dict = {},
):
    if not metric:
        raise ValueError("Metric is None or empty.")

    if metric_df is None or metric_df.empty:
        raise ValueError("Metric dataframe is None.")

    if pairings_df is None or pairings_df.empty:
        raise ValueError("Pairings dataframe is None.")

    if not param_variations_dict_list:
        raise ValueError("Parameter variations list is None or empty.")

    if not sections:
        raise ValueError("Sections list is None or empty.")

    if not exp_variation_dict:
        raise ValueError("Experiment variation dictionary is None or empty.")

    pairings_df["value_a"] = pairings_df["value_a"].astype(str)
    pairings_df["value_b"] = pairings_df["value_b"].astype(str)

    pairing_values_df = pairings_df.copy()
    pairing_values_df = (
        pairing_values_df[["value_a", "value_b"]]
        .drop_duplicates(subset=["value_a", "value_b"])
        .reset_index(drop=True)
    )

    pairing_titles = get_pairing_title_from_df(pairing_values_df)
    pairings_count = len(pairing_values_df)
    sec_count = len(sections)

    fig, axs = plt.subplots(
        nrows=pairings_count + 1,  # Extra row for the most performant line
        ncols=sec_count,
        figsize=(5 * sec_count, 5 * pairings_count),
    )

    for pairing_i, pairing_df_row in pairing_values_df.iterrows():
        # Get the pairings for the current pairing
        value_a = pairing_df_row["value_a"]
        value_b = pairing_df_row["value_b"]

        a_title = format_param_dict_to_title(ast.literal_eval(value_a))
        b_title = format_param_dict_to_title(ast.literal_eval(value_b))

        legend_titles = pairing_titles.copy()

        # Remove a_title and b_title from the pairing_titles
        # to add them later to the legend but with colours
        legend_titles.remove(a_title)
        legend_titles.remove(b_title)

        pairing_df = pairings_df[
            (pairings_df["value_a"] == value_a) & (pairings_df["value_b"] == value_b)
        ]
        pairing_df = pairing_df.reset_index(drop=True)

        assert pairing_df["value_a"].nunique() == 1
        assert pairing_df["value_b"].nunique() == 1

        for sec_i, section in enumerate(sections):
            # Get the section for the current pairing
            sec_df = pairing_df[
                (pairing_df["probability"] >= section["start"])
                & (pairing_df["probability"] <= section["end"])
            ]
            sec_df = sec_df.reset_index(drop=True)

            a_df = sec_df[(sec_df["value_a"] == value_a)]
            a_n = len(a_df)

            ax = axs[pairing_i, sec_i]
            ax = draw_section_lines(ax, section)

            ax = plot_all_cdfs(pairings_df, ax)
            ax, color_a, color_b = plot_focused_cdf(metric, sec_df, ax)

            if pairing_i == 0:
                ax.set_title(f"{section['start']} - {section['end']}, n={a_n:,}")
            else:
                ax.set_title(f"n={a_n:,}")

            if a_n == 0:
                color_a = "#aaa"
                color_b = "#aaa"

            ax = decorate_focused_ax(ax, section, metric)

            legend_handles = []
            for title in pairing_titles:
                legend_handles.append(
                    mpatches.Patch(color="#aaa", alpha=0.1, label=title)
                )

            ax_legend = ax.legend(handles=legend_handles)

            for title in pairing_titles:
                color_legend_text(ax_legend, title, "#aaa")
            color_legend_text(ax_legend, a_title, color_a)
            color_legend_text(ax_legend, b_title, color_b)

    axs = plot_most_performant_cdfs(axs, pairings_df, metric, sections, pairing_titles)

    plt.tight_layout()
    exp_variation_str = abbreviate_param_names(
        format_dict(exp_variation_dict, separator="-")
    )
    parameters = param_variations_dict_list[0].keys()
    parameter_str = " vs ".join(parameters)

    for extension in ["png"]:
        filepath = "./output/parameter_analysis_tool/{}/working_out_{}/{}.{}".format(
            parameter_str, metric, exp_variation_str, extension
        )

        plt.savefig(filepath)

    plt.close()


def plot_most_performant_cdfs(axs, pairings_df, metric, sections, pairing_titles):
    """
    Group into a vs b. For each a vs b, find the mode most performant line.
    """

    for sec_i, section in enumerate(sections):
        ax = axs[-1, sec_i]

        sec_df = pairings_df[
            (pairings_df["probability"] >= section["start"])
            & (pairings_df["probability"] <= section["end"])
        ]
        sec_df = sec_df.reset_index(drop=True)

        if len(sec_df) == 0:
            continue

        ax = plot_all_cdfs(pairings_df, ax)

        most_perf_param_dict = get_most_perf_param_dict(sec_df, metric)

        no_winner = False
        ax_title = "No Winner"
        if not most_perf_param_dict:
            no_winner = True

        value_a_vals = sec_df["value_a"].unique()
        value_b_vals = sec_df["value_b"].unique()

        if most_perf_param_dict in value_a_vals:
            letter = "a"
        elif most_perf_param_dict in value_b_vals:
            letter = "b"
        else:
            no_winner = True

        if not no_winner:
            lg.info(
                f"\t[{section['start']} - {section['end']}] Winner: {most_perf_param_dict}"
            )

            winner_df = sec_df[sec_df[f"value_{letter}"] == most_perf_param_dict]
            wanted_cols = [col for col in sec_df.columns if f"_{letter}" in col]
            wanted_cols.append("probability")
            winner_df = winner_df[wanted_cols]
            ax = plot_single_cdf_with_conf(winner_df, letter, "#488f31", ax)
            winner_n = len(winner_df)

            winner_title = format_param_dict_to_title(
                ast.literal_eval(most_perf_param_dict)
            )
            winner_title = abbreviate_param_names(winner_title)
            ax_title = f"{winner_title} (n={winner_n:,})"

        ax = draw_section_lines(ax, section)
        ax = decorate_focused_ax(ax, section, metric)
        ax = shade_out_of_section(ax, section)

        if not no_winner:
            ax.spines["left"].set_color("#488f31")
            ax.spines["bottom"].set_color("#488f31")

        legend_handles = []
        for title in pairing_titles:
            legend_handles.append(mpatches.Patch(color="#aaa", alpha=0.1, label=title))

        ax_legend = ax.legend(handles=legend_handles)

        ax.set_title(ax_title)
        ax.set_facecolor((0.28, 0.56, 0.19, 0.1))

        for title in pairing_titles:
            color_legend_text(ax_legend, title, "#aaa")

        if not no_winner:
            color_legend_text(ax_legend, winner_title, "#488f31")

    return axs


def format_metric_title(metric):
    if metric == "latency_us":
        return "Latency (µs)"

    elif metric == "total_mbps":
        return "Total Throughput (Mbps)"

    else:
        return metric


def plot_focused_cdf(metric, sec_df, ax):
    assert sec_df["value_a"].nunique() <= 1
    assert sec_df["value_b"].nunique() <= 1

    if len(sec_df) == 0:
        return ax, "#aaa", "#aaa"

    all_a_on_left = all(sec_df["a_on_left"])
    all_b_on_left = all(sec_df["b_on_left"])

    color_a = "#aaa"
    color_b = "#aaa"

    """
    a left + b left = Impossible - how can they both be left?
    a left + b not left = a is left
    a not left + b left = b is left
    a not left + b not left = Might be intersecting
    """

    # a left + b left = Impossible - how can they both be left?
    assert not (all_b_on_left & all_a_on_left)

    if all_a_on_left and not all_b_on_left:
        if metric == "latency_us":
            color_a = "#488f31"
            color_b = "#de425b"
        else:
            color_a = "#de425b"
            color_b = "#488f31"

    elif not all_a_on_left and all_b_on_left:
        if metric == "latency_us":
            color_a = "#de425b"
            color_b = "#488f31"
        else:
            color_a = "#488f31"
            color_b = "#de425b"

    elif not all_a_on_left and not all_b_on_left:
        color_a = "#003f5c"
        color_b = "#ffa600"

    else:
        raise ValueError(
            f"Invalid combination of a_on_left and b_on_left: a_left={all_a_on_left}, b_left={all_b_on_left}"
        )

    plot_single_cdf_with_conf(sec_df, "a", color_a, ax)
    plot_single_cdf_with_conf(sec_df, "b", color_b, ax)

    return ax, color_a, color_b


def shade_out_of_section(ax, section):
    ranges = [(section["end"], 1), (0, section["start"])]

    for range in ranges:
        ax.fill_between(
            ax.get_xlim(),
            range[0],
            range[1],
            color="none",
            alpha=0.1,
            hatch="//",
            edgecolor="#000",
            linewidth=0,
        )

    return ax


def plot_single_cdf_with_conf(sec_df, letter, color, ax):
    ax.scatter(
        sec_df[f"quantile_{letter}"],
        sec_df["probability"],
        color=color,
        s=5,
        alpha=1,
        marker="x",
    )

    for col in ["low", "high"]:
        ax.scatter(
            sec_df[f"quantile_{letter}_{col}"],
            sec_df["probability"],
            color=color,
            s=10,
            alpha=1,
            marker="|",
        )

    ax.fill_betweenx(
        sec_df["probability"],
        sec_df[f"quantile_{letter}_low"],
        sec_df[f"quantile_{letter}_high"],
        color=color,
        alpha=0.1,
    )

    return ax


def plot_all_cdfs(pairings_df, ax):
    ax.scatter(
        pairings_df["quantile_a"],
        pairings_df["probability"],
        color="#000",
        s=1,
        alpha=0.2,
    )

    ax.scatter(
        pairings_df["quantile_b"],
        pairings_df["probability"],
        color="#000",
        s=1,
        alpha=0.2,
    )

    return ax


def draw_section_lines(ax, section):
    ax.axhline(y=section["start"], color="black", linestyle="dashed", alpha=0.1)
    ax.axhline(y=section["end"], color="black", linestyle="dashed", alpha=0.1)

    # Add tick labels to y-axis
    current_yticks = ax.get_yticks()
    new_yticks = list(current_yticks) + [section["start"], section["end"]]
    ax.set_yticks(new_yticks)

    return ax


def get_pairing_title_from_df(pairing_values_df):
    pairing_titles = []

    for pairing_i, pairing_df_row in pairing_values_df.iterrows():
        value_a = pairing_df_row["value_a"]
        value_b = pairing_df_row["value_b"]

        a_title = format_param_dict_to_title(ast.literal_eval(value_a))
        b_title = format_param_dict_to_title(ast.literal_eval(value_b))

        pairing_titles.append(a_title)
        pairing_titles.append(b_title)

    pairing_titles = list(set(pairing_titles))

    return pairing_titles


def format_param_dict_to_title(param_dict: dict):
    if not param_dict:
        raise ValueError("Parameter dictionary is None or empty.")

    if not isinstance(param_dict, dict):
        if isinstance(param_dict, str):
            try:
                param_dict = ast.literal_eval(param_dict)
            except Exception as e:
                raise ValueError(f"Could not convert {param_dict} to dict:\n\t{e}")

        else:
            raise TypeError("Parameter dictionary is not a dictionary.")

    param_str = format_dict(param_dict)
    param_str = abbreviate_param_names(param_str)
    return param_str


def abbreviate_param_names(param_str: str):
    replacements = {
        "datalen_bytes": "Data(B)",
        "pub_count": "Pubs",
        "sub_count": "Subs",
        "use_reliable": "Rel",
        "use_multicast": "MC",
        "durability": "Dur",
    }

    for key, value in replacements.items():
        param_str = param_str.replace(key, value)

    return param_str


def process_pairing_df(
    metric: str = "latency_us",
    pair_a_dict: dict = {},
    pair_b_dict: dict = {},
    metric_df: pd.DataFrame = pd.DataFrame(),
):
    """
    Does the following:
        - Gets the latency/throughput data for pair A and pair B
        - Calculates the ECDF for pair A and pair B
        - Calculates the error margin for pair A and pair B
        - Merges the ECDFs for pair A and pair B on the y-values
        - Calculates the difference between the quantiles of pair A and pair B
          for intersection purposes
        - Identifies if pair A is on the left and if pair B is on the left

    Args:
        - metric: The metric to process
        - pair_a_dict: The dictionary of pair A
        - pair_b_dict: The dictionary of pair B
        - metric_df: The dataframe containing the latency/throughput data

    Returns:
        - a_b_df: The ECDFs of pair A and pair B
        - metric_a_df: The latency/throughput data for pair A
        - metric_b_df: The latency/throughput data for pair B
    """
    if not metric:
        raise ValueError("Metric is None or empty.")

    if not pair_a_dict or not pair_b_dict:
        raise ValueError("Pairing dictionaries are None or empty.")

    if metric_df.empty:
        raise ValueError("Metric dataframe is None or empty.")

    # Create boolean masks for filtering
    mask_a = pd.Series(True, index=metric_df.index)
    for key, value in pair_a_dict.items():
        mask_a &= metric_df[key] == value

    mask_b = pd.Series(True, index=metric_df.index)
    for key, value in pair_b_dict.items():
        mask_b &= metric_df[key] == value

    metric_a_df = metric_df.loc[mask_a, metric].dropna().reset_index(drop=True)
    metric_b_df = metric_df.loc[mask_b, metric].dropna().reset_index(drop=True)

    if metric_a_df.empty or metric_b_df.empty:
        return None, None, None

    res_a = scistats.ecdf(metric_a_df)
    res_b = scistats.ecdf(metric_b_df)

    error_margin_a = calculate_error_margin(metric_a_df)
    error_margin_b = calculate_error_margin(metric_b_df)

    cdf_a_df = (
        pd.DataFrame(
            {
                "quantile_a": list(res_a.cdf.quantiles),
                "probability": list(res_a.cdf.probabilities),
                "error_margin_a": error_margin_a,
            }
        )
        .round({"probability": 2})
        .drop_duplicates("probability", keep="first")
    )

    cdf_b_df = (
        pd.DataFrame(
            {
                "quantile_b": list(res_b.cdf.quantiles),
                "probability": list(res_b.cdf.probabilities),
                "error_margin_b": error_margin_b,
            }
        )
        .round({"probability": 2})
        .drop_duplicates("probability", keep="first")
    )

    a_b_df = pd.merge(cdf_a_df, cdf_b_df, on="probability", how="outer")
    a_b_df[["quantile_a", "quantile_b"]] = a_b_df[
        ["quantile_a", "quantile_b"]
    ].interpolate()
    a_b_df["error_margin_a"] = error_margin_a
    a_b_df["error_margin_b"] = error_margin_b
    a_b_df.dropna(subset=["quantile_a", "quantile_b"], inplace=True)

    for x in ["a", "b"]:
        a_b_df[f"quantile_{x}_low"] = (
            a_b_df[f"quantile_{x}"] - a_b_df[f"error_margin_{x}"]
        )
        a_b_df[f"quantile_{x}_high"] = (
            a_b_df[f"quantile_{x}"] + a_b_df[f"error_margin_{x}"]
        )

    a_b_df["b_low_a_high_diff"] = a_b_df["quantile_b_low"] - a_b_df["quantile_a_high"]
    a_b_df["a_low_b_high_diff"] = a_b_df["quantile_a_low"] - a_b_df["quantile_b_high"]
    a_b_df["a_on_left"] = a_b_df["b_low_a_high_diff"] > 0
    a_b_df["b_on_left"] = a_b_df["a_low_b_high_diff"] > 0

    return a_b_df, metric_a_df, metric_b_df


def decorate_focused_ax(ax, section, metric):
    ax = shade_out_of_section(ax, section)

    ax.set_ylim(section["start"] - 0.1, section["end"] + 0.1)

    ax.set_yticks([section["start"], section["end"]])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(format_metric_title(metric))
    ax.set_ylabel("CDF")

    return ax


def format_dict(dictionary, separator="="):
    if not dictionary:
        raise ValueError("No dict passed to format_dict().")

    if not isinstance(dictionary, dict):
        # Try to convert to dict
        if isinstance(dictionary, str):
            try:
                dictionary = ast.literal_eval(dictionary)

            except Exception as e:
                raise e

        else:
            raise ValueError(
                f"Dict passed to format_dict() is not a dict:\n\t{dictionary}"
            )

    return " ".join([f"{key}{separator}{value}" for key, value in dictionary.items()])


def filter_df_by_dict(df, dict):
    if not dict:
        lg.error("No dict passed.")
        return df

    if len(dict) == 0:
        lg.error("No filtering required.")
        return df

    for key, value in dict.items():
        if key not in df.columns:
            raise ValueError(f"Key {key} not in dataframe.")

    df_copy = df.copy()

    for key, value in dict.items():
        df_copy = df_copy[df_copy[key] == value]

    return df_copy


def calculate_error_margin(df, z=0.95):
    """
    Calculates the error margin for a given dataframe using the formula:
        margin = z * (std / sqrt(n))

    Args:
        - df: The dataframe to calculate the error margin for
        - z: The z-value for the confidence interval

    Returns:
        - margin: The error margin
    """

    if df is None:
        raise ValueError("No dataframe passed to calculate_error_margin().")

    if len(df) == 0:
        raise ValueError("Empty dataframe passed to calculate_error_margin().")

    if z <= 0 or z >= 1:
        raise ValueError("Invalid z-value passed to calculate_error_margin().")

    std = np.std(df)
    n = len(df)
    margin = z * (std / sqrt(n))

    return margin


def make_needed_folders(param_str, metric):
    folder_paths = [
        f"./output/parameter_analysis_tool/{param_str}",
        f"./output/parameter_analysis_tool/{param_str}/pairings",
        f"./output/parameter_analysis_tool/{param_str}/variations",
        f"./output/parameter_analysis_tool/{param_str}/working_out_{metric}",
        "./output/parameter_analysis_tool/classification_explorer",
        f"./output/parameter_analysis_tool/classification_explorer/{param_str}",
    ]

    for folder_path in folder_paths:
        os.makedirs(folder_path, exist_ok=True)


def process_all_pairings(
    variations_df: pd.DataFrame = pd.DataFrame(),
    exp_variation_dict_list: list = [],
    varied_param_list: list = [],
    metric_cols: list = [],
    sections: list = [],
):
    param_str = " vs ".join(varied_param_list)
    all_pairings_df_filepath = (
        f"./output/parameter_analysis_tool/{param_str}/all_pairings.parquet"
    )

    if not d_config["overwrite_processing"]["all_pairings"] and os.path.exists(
        all_pairings_df_filepath
    ):
        all_pairings_df = pd.read_parquet(all_pairings_df_filepath)
        lg.info(
            f"Loaded {len(all_pairings_df)} pairings from {all_pairings_df_filepath}..."
        )
        return all_pairings_df

    if not exp_variation_dict_list:
        lg.warning(
            "exp_variation_dict_list is empty. Cannot determine unvaried parameters."
        )
        return pd.DataFrame()

    unvaried_params = list(exp_variation_dict_list[0].keys())

    grouped_variations = variations_df.groupby(unvaried_params)
    all_pairings_list = []

    for i, (group_keys, group_df) in enumerate(grouped_variations):
        if isinstance(group_keys, tuple):
            exp_variation_dict = dict(zip(unvaried_params, group_keys))
        else:
            exp_variation_dict = {unvaried_params[0]: group_keys}

        # Cast all np types to native python types for consistency
        for key, value in exp_variation_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                exp_variation_dict[key] = value.item()
            elif isinstance(value, np.bool_):
                exp_variation_dict[key] = bool(value)

        exp_variation_str = format_param_dict_to_title(exp_variation_dict)
        lg.info(f"[{i+1}/{len(grouped_variations)}] {exp_variation_str}")

        param_variations_dict_list = (
            group_df[varied_param_list].drop_duplicates().to_dict(orient="records")
        )
        param_variations_dict_list = sort_list_of_dicts(param_variations_dict_list)

        if len(param_variations_dict_list) <= 1:
            lg.warning(
                f"{exp_variation_str} has {len(param_variations_dict_list)} variations, not enough for pairwise comparison."
            )
            continue

        for metric in metric_cols:
            status_prefix = f"[{i+1}/{len(grouped_variations)}] [{exp_variation_str}] [{metric.upper()}]"

            metric_df = (
                group_df[[metric] + varied_param_list]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if metric_df.empty:
                lg.error(f"{status_prefix} No data found for {metric}.")
                continue

            lg.info(f"{status_prefix} Processing pairings...")
            pairings_df = process_pairings(
                param_variations_dict_list, metric, metric_df
            )

            if pairings_df is None or pairings_df.empty:
                lg.error(f"{status_prefix} Pairings df came back empty.")
                continue

            pairings_df["exp_variation"] = str(exp_variation_dict)
            all_pairings_list.append(pairings_df)

            lg.info(f"{status_prefix} Plotting working out...")
            if len(param_variations_dict_list) < 10:
                show_working_out(
                    metric,
                    metric_df,
                    pairings_df,
                    param_variations_dict_list,
                    sections,
                    exp_variation_dict,
                )
            else:
                lg.warning(
                    f"{exp_variation_str} has {len(param_variations_dict_list)} variations. Skipping plotting..."
                )

    if not all_pairings_list:
        return pd.DataFrame()

    all_pairings_df = pd.concat(all_pairings_list, ignore_index=True)

    write_df(all_pairings_df, all_pairings_df_filepath)

    return all_pairings_df


def classify_pairings(
    all_pairings_df: pd.DataFrame = pd.DataFrame(),
    varied_param_list: list = [],
    d_config: dict = {},
):
    if all_pairings_df is None or all_pairings_df.empty:
        raise ValueError("No pairings found.")

    all_pairings_df = all_pairings_df.dropna().reset_index(drop=True)

    assert "metric" in all_pairings_df.columns
    assert all_pairings_df["metric"].nunique() == len(
        d_config["metric_columns"]
    ), "metrics in df: {}\nmetrics in config: {}".format(
        all_pairings_df["metric"].unique(), d_config["metric_columns"]
    )

    most_perf_pairings_df = get_most_perf_pairings(
        all_pairings_df, varied_param_list, d_config
    )

    if most_perf_pairings_df is None or most_perf_pairings_df.empty:
        raise ValueError("No most performant pairings found.")

    unique_possible_values = most_perf_pairings_df["possible_values"].unique()
    unique_possible_values = [ast.literal_eval(val) for val in unique_possible_values]
    unique_possible_values = sorted(unique_possible_values, key=lambda x: len(x))

    for possible_values in unique_possible_values:
        lg.info(
            "Classifying pairings for {}...".format(
                [format_param_dict_to_title(val) for val in possible_values]
            )
        )

        possible_values_df = most_perf_pairings_df.copy()
        possible_values_df = possible_values_df[
            possible_values_df["possible_values"] == str(possible_values)
        ]

        print_most_perf_pairings(
            possible_values_df,
            varied_param_list,
            "all_sections",
            possible_values,
            d_config,
        )

    return most_perf_pairings_df


def get_unvaried_param_list(
    var_df: pd.DataFrame = pd.DataFrame(),
) -> tuple[list, pd.DataFrame]:
    params = []
    for var_i, var_df_row in var_df.iterrows():
        param_dict = ast.literal_eval(var_df_row["exp_variation"])

        for key, value in param_dict.items():
            if isinstance(value, str) and "error" in value:
                value_str = "Uncategorised"
            else:
                value_str = value

            var_df.loc[var_i, key] = value_str

            if key not in params:
                params.append(key)

        if "error" in var_df_row["most_perf_param_dict"]:
            var_df.loc[var_i, "most_perf_param_dict"] = "Uncategorised"
        else:
            var_df.loc[var_i, "most_perf_param_dict"] = format_dict(
                var_df_row["most_perf_param_dict"]
            )

    return params, var_df


def get_chi_square_results_df(
    section_df: pd.DataFrame = pd.DataFrame(),
    correlation_df: pd.DataFrame = pd.DataFrame(),
):
    if section_df.empty:
        raise ValueError("Section dataframe is empty.")

    unvaried_param_list, section_df = get_unvaried_param_list(section_df)

    for param in unvaried_param_list:
        if param not in section_df.columns:
            raise ValueError(f"Parameter {param} not in section dataframe.")

        section_df[param] = section_df[param].astype(str)

    chi_square_df = get_chi_square_df(section_df, unvaried_param_list)

    correlation_df = pd.concat(
        [correlation_df, chi_square_df], ignore_index=True, axis=1
    )

    return correlation_df


def get_unique_possible_values(df: pd.DataFrame = pd.DataFrame()):
    if df.empty:
        raise ValueError("Dataframe is empty.")

    unique_possible_values = df["possible_values"].unique()
    unique_possible_values = [
        ast.literal_eval(value) for value in unique_possible_values
    ]
    unique_possible_values = sorted(unique_possible_values, key=lambda x: len(x))

    return unique_possible_values


def populate_cramers_v_df_col_names(
    cramers_v_df: pd.DataFrame = pd.DataFrame(), d_config: dict = {}
):
    cramers_v_df_col_count = len(cramers_v_df.columns)
    # Rename index to Parameter
    cramers_v_df.index.name = "Parameter"

    column_names = []
    for section in d_config["sections"][:cramers_v_df_col_count]:
        section_start = section["start"]
        section_end = section["end"]

        column_names.append(
            f"{int(section_start * 100)} to {int(section_end * 100)} Cramers V"
        )

    cramers_v_df.columns = column_names

    for col in cramers_v_df.columns:
        if "Count" in col:
            cramers_v_df[col] = cramers_v_df[col].apply(
                lambda x: str(int(x)) if x != "" else ""
            )

    return cramers_v_df


def populate_chi_square_df_col_names(
    correlation_df: pd.DataFrame = pd.DataFrame(), d_config: dict = {}
):
    correlation_col_count = len(correlation_df.columns)

    column_names = []
    for section in d_config["sections"][: correlation_col_count // 3]:
        section_start = section["start"]
        section_end = section["end"]

        cols_per_param = ["Chi-Square", "P-Value", "DoF"]
        for col in cols_per_param:
            column_names.append(
                f"{int(section_start * 100)} to {int(section_end * 100)} {col}"
            )

    correlation_df.columns = column_names

    for col in correlation_df.columns:
        if "Count" in col:
            correlation_df[col] = correlation_df[col].apply(
                lambda x: float(x) if x != "" else 0.0
            )

    return correlation_df


def populate_classification_col_names(
    classification_df: pd.DataFrame = pd.DataFrame(), d_config: dict = {}
):
    classification_col_count = len(classification_df.columns)

    column_names = []
    for section in d_config["sections"][:classification_col_count]:
        section_start = section["start"]
        section_end = section["end"]

        column_names.append(
            f"{int(section_start * 100)} to {int(section_end * 100)} Count"
        )

    classification_df.columns = column_names

    for col in classification_df.columns:
        if "Count" in col:
            classification_df[col] = classification_df[col].apply(
                lambda x: int(x) if x != "" else 0
            )

    return classification_df


def create_results_report(
    metric: str = "latency_us",
    most_perf_pairings_df: pd.DataFrame = pd.DataFrame(),
    varied_param_list: list = [],
    d_config: dict = {},
):
    if most_perf_pairings_df.empty:
        raise ValueError("Most performant pairings dataframe is empty.")

    if len(varied_param_list) == 0:
        raise ValueError("No varied parameter list passed to create_results_report().")

    metric_df = most_perf_pairings_df.copy()
    metric_df = metric_df[metric_df["metric"] == metric].reset_index(drop=True)

    report_content = ""

    param_str = " vs ".join(varied_param_list)
    output_dir = f"./output/parameter_analysis_tool/classification_explorer/{param_str}"
    report_filename = f"{metric}_classification_report.md"
    os.makedirs(output_dir, exist_ok=True)

    unique_possible_values = get_unique_possible_values(most_perf_pairings_df)

    for possible_vals in unique_possible_values:
        possible_vals_str_list = [
            format_param_dict_to_title(val) for val in possible_vals
        ]
        report_content += f"## {" vs ".join(possible_vals_str_list)}\n\n"

        possible_vals_df = metric_df.copy()
        possible_vals_df = possible_vals_df[
            possible_vals_df["possible_values"] == str(possible_vals)
        ].reset_index(drop=True)

        classification_df = pd.DataFrame()
        chi_square_df = pd.DataFrame()
        cramers_v_df = pd.DataFrame()
        for section in d_config["sections"]:
            section_start = section["start"]
            section_end = section["end"]

            section_df = possible_vals_df.copy()
            section_df = section_df[
                (section_df["section_start"] == section_start)
                & (section_df["section_end"] == section_end)
            ].reset_index(drop=True)

            if section_df.empty:
                continue

            classification_df = get_classification_df(
                section_df, section_start, section_end, classification_df
            )

            chi_square_df = get_chi_square_results_df(section_df, chi_square_df)

            cramers_v_df = get_cramers_v_results_df(section_df, cramers_v_df)

        classification_df.replace(np.nan, "", inplace=True)
        classification_df = populate_classification_col_names(
            classification_df, d_config
        )
        classification_df.sort_index(inplace=True)
        classification_df = add_total_row_to_df(classification_df)

        chi_square_df.replace(np.nan, "", inplace=True)
        chi_square_df = populate_chi_square_df_col_names(chi_square_df, d_config)

        cramers_v_df.replace(np.nan, "", inplace=True)
        cramers_v_df = populate_cramers_v_df_col_names(cramers_v_df, d_config)

        for df in [classification_df, chi_square_df, cramers_v_df]:
            report_content += df.to_markdown(index=True)
            report_content += "\n\n"

    with open(f"{output_dir}/{report_filename}", "w") as f:
        f.write(report_content)

    lg.info(f"Results report saved to {output_dir}/{report_filename}.")


def plot_classification_bar_chart(
    classification_df: pd.DataFrame = pd.DataFrame(), filepath: str = ""
):
    if classification_df.empty:
        raise ValueError("Classification dataframe is empty.")

    plot_df = classification_df.copy()

    df_melted = plot_df.melt(
        id_vars="Parameter", var_name="Section", value_name="Count"
    )

    # Remove all rows with Total in Paraemter
    df_melted = df_melted[~df_melted["Parameter"].str.contains("Total")]

    fig, ax = plt.subplots(figsize=(10, 5))

    custom_colors = [
        "#003f5c",
        "#444e86",
        "#dd5182",
        "#955196",
        "#ff6e54",
        "#ffa600",
    ]

    ax = sns.barplot(
        data=df_melted,
        x="Section",
        y="Count",
        hue="Parameter",
        palette=custom_colors,
        ax=ax,
    )

    ax.set_ylabel("Count")
    ax.set_xlabel("Parameter")

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(title="Section", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    lg.info(f"Classification bar chart saved to {filepath}.")


def process_exp_variation_into_cols(section_pairings_df: pd.DataFrame = pd.DataFrame()):
    if section_pairings_df.empty:
        lg.error("Section pairings dataframe is empty.")
        return section_pairings_df

    for df_i, df_row in section_pairings_df.iterrows():
        exp_var_str = df_row["exp_variation"]
        exp_var_dict = ast.literal_eval(exp_var_str)

        for key, value in exp_var_dict.items():
            section_pairings_df.loc[df_i, key] = value

    return section_pairings_df


def plot_param_distribution_per_section(
    df: pd.DataFrame = pd.DataFrame(), output_dir: str = ""
):
    custom_palette = [
        "#488f31",
        "#de425b",
        "#003f5c",
        "#955196",
        "#ff7c43",
    ]

    plot_df = df.copy()

    section_title = df["section_title"].iloc[0]

    output_path = f"{output_dir}/{section_title} histogram.png"
    os.makedirs(output_dir, exist_ok=True)

    exp_param_cols = [col for col in df.columns if col in PARAM_COLS]

    fig, axs = plt.subplots(
        nrows=1, ncols=len(exp_param_cols), figsize=(5 * len(exp_param_cols), 5)
    )

    plot_df["most_perf_param_dict"] = plot_df["most_perf_param_dict"].apply(
        lambda x: (
            format_param_dict_to_title(ast.literal_eval(x))
            if "error" not in x
            else "Uncategorised"
        )
    )

    # Rename most_perf_param_dict to Classification
    plot_df.rename(columns={"most_perf_param_dict": "Classification"}, inplace=True)

    for col_i, col in enumerate(exp_param_cols):
        ax = axs[col_i]

        sns.histplot(
            data=plot_df,
            x=col,
            hue="Classification",
            ax=ax,
            multiple="dodge",
            palette=custom_palette,
        )

        ax.set_ylabel("Count")

        ax.grid(axis="y", linestyle="--", alpha=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path)
    lg.info(
        f"Parameter distribution plot for section {section_title} saved to {output_path}."
    )


def plot_param_distribution(
    met_df: pd.DataFrame = pd.DataFrame(),
    varied_param_list: list = [],
    possible_vals_dict_list: list = [],
    metric_col: str = "",
):
    output_dir = "./output/parameter_analysis_tool/classification_explorer"
    param_str = " vs ".join(varied_param_list)
    possible_vals_list = [
        format_param_dict_to_title(val) for val in possible_vals_dict_list
    ]
    possible_vals_str = "_vs_".join(possible_vals_list).replace("=", "-")

    output_dir = f"{output_dir}/{param_str}/{possible_vals_str}/{metric_col}"

    for col in ["section_start", "section_end"]:
        met_df[col] = met_df[col].apply(lambda x: str(int(x * 100)))

    met_df["section_title"] = met_df["section_start"] + " to " + met_df["section_end"]

    met_df = met_df.groupby("section_title").apply(
        lambda x: plot_param_distribution_per_section(x, output_dir)
    )


def print_most_perf_pairings(
    most_perf_pairings_df: pd.DataFrame = pd.DataFrame(),
    varied_param_list: list = [],
    section_title: str = "all_sections",
    possible_values: list = [],
    d_config: dict = {},
):
    for metric_col in d_config["metric_columns"]:
        metric_pairings_df = most_perf_pairings_df.copy()
        metric_pairings_df = metric_pairings_df[
            most_perf_pairings_df["metric"] == metric_col
        ].reset_index(drop=True)

        metric_pairings_df = process_exp_variation_into_cols(metric_pairings_df)
        plot_param_distribution(
            metric_pairings_df, varied_param_list, possible_values, metric_col
        )

        results_df = pd.DataFrame()
        for section in d_config["sections"]:
            section_start = section["start"]
            section_end = section["end"]

            section_start_str = str(int(section_start * 100))
            section_end_str = str(int(section_end * 100))

            section_pairings_df = metric_pairings_df.copy()
            section_pairings_df = section_pairings_df[
                (metric_pairings_df["section_start"] == section_start)
                & (metric_pairings_df["section_end"] == section_end)
            ].reset_index(drop=True)

            if section_pairings_df.empty:
                continue

            results = section_pairings_df["most_perf_param_dict"]
            results = results.value_counts().to_dict()

            sec_results_df = pd.DataFrame(results, index=[0]).T
            sec_results_df.columns = [f"{section_start_str} to {section_end_str} Count"]
            sec_results_df.index.name = "Parameter Index"

            results_df = pd.concat([results_df, sec_results_df], axis=1)

        results_df["Parameter"] = results_df.index.map(str)
        results_df["Parameter"] = results_df["Parameter"].apply(
            lambda x: (
                format_param_dict_to_title(x) if "error" not in x else "Uncategorised"
            )
        )
        results_df = results_df.sort_values(by="Parameter")

        total_row = results_df.sum(axis=0).to_frame().T
        total_row["Parameter"] = ["Total"]
        results_df = pd.concat([results_df, total_row], axis=0)

        results_df = results_df.fillna(0)
        count_cols = [col for col in results_df.columns if "Count" in col]
        for col in count_cols:
            results_df[col] = results_df[col].astype(int)

        new_col_order = ["Parameter"] + count_cols
        results_df = results_df[new_col_order]
        results_df = results_df.reset_index(drop=True)

        if varied_param_list == []:
            param_str = "All"
        else:
            param_str = " vs ".join(varied_param_list)

        possible_values_str = [
            format_param_dict_to_title(val) for val in possible_values
        ]
        possible_values_str = "_vs_".join(possible_values_str)
        possible_values_str = possible_values_str.replace("=", "-")

        md_filedir = "./output/parameter_analysis_tool/classification_explorer"
        md_filedir = f"{md_filedir}/{param_str}"
        md_filepath = "{}/{}/{}/{}.md".format(
            md_filedir, possible_values_str, metric_col, section_title
        )

        os.makedirs(os.path.dirname(md_filepath), exist_ok=True)
        os.makedirs(f"{md_filedir}/{metric_col}", exist_ok=True)

        with open(md_filepath, "w") as f:
            f.write(results_df.to_markdown(index=False))

        for ext in ["png", "pdf"]:
            plot_classification_bar_chart(
                results_df,
                "./{}/{}".format(
                    os.path.dirname(md_filepath), f"classifications_per_section.{ext}"
                ),
            )

        lg.info(
            "Results for {} saved to\n\t{}/{}.md".format(
                metric_col, param_str, md_filepath.replace("/", "/\n\t")
            )
        )

        # Print the table
        # console.print(results_table)


def get_most_perf_pairings(
    all_pairings_df: pd.DataFrame = pd.DataFrame(),
    varied_param_list: list = [],
    d_config: dict = {},
):
    if not varied_param_list:
        raise ValueError("No varied parameter list passed to get_most_perf_pairings().")

    param_str = " vs ".join(varied_param_list)
    most_perf_pairings_filename = "most_perf_pairings.parquet"
    most_perf_pairings_filepath = "./output/parameter_analysis_tool/{}/{}".format(
        param_str, most_perf_pairings_filename
    )

    if not d_config["overwrite_processing"]["most_perf_pairings"] and os.path.exists(
        most_perf_pairings_filepath
    ):
        lg.info("Loading most performant pairings...")
        most_perf_pairings_df = pd.read_parquet(most_perf_pairings_filepath)

    else:
        most_perf_pairings_df = process_most_performant_pairings(
            all_pairings_df, d_config
        )

        most_perf_pairings_df.to_parquet(most_perf_pairings_filepath, index=False)

    return most_perf_pairings_df


def get_most_perf_pairing_per_exp_var(
    df: pd.DataFrame = pd.DataFrame(), metric: str = ""
):
    if df is None or df.empty:
        raise ValueError("No dataframe passed to get_most_perf_pairing_per_exp_var().")

    for col in ["value_a", "value_b"]:
        df[col] = df[col].astype(str)

    unique_pairings_df = df.copy()
    unique_pairings_df = unique_pairings_df.drop_duplicates(
        subset=["value_a", "value_b"]
    ).reset_index(drop=True)

    most_perf_param_dict = get_most_perf_param_dict(df, metric)
    if not most_perf_param_dict:
        most_perf_param_dict = {"error": "Uncategorised"}
    else:
        most_perf_param_dict = ast.literal_eval(most_perf_param_dict)

    possible_values = df["possible_values"].iloc[0]

    most_perf_pairings_df = pd.DataFrame(
        {
            "exp_variation": [df["exp_variation"].iloc[0]],
            "most_perf_param_dict": str(most_perf_param_dict),
            "metric": [metric],
            "possible_values": [possible_values],
        }
    )

    return most_perf_pairings_df


def process_most_performant_pairings(
    all_pairings_df: pd.DataFrame = pd.DataFrame(), d_config: dict = {}
):
    if all_pairings_df.empty:
        raise ValueError("No pairings found.")

    if "possible_values" not in all_pairings_df.columns:
        raise ValueError("No possible_values column found in all_pairings_df.")

    most_perf_pairings_df = pd.DataFrame()

    for metric_col_i, metric in enumerate(d_config["metric_columns"]):
        metric_col_i_str = (
            f"[{metric_col_i + 1}/{len(d_config['metric_columns'])} {metric.upper()}]"
        )

        metric_pairings_df = all_pairings_df.copy()
        metric_pairings_df = metric_pairings_df[
            all_pairings_df["metric"] == metric
        ].reset_index(drop=True)

        if metric_pairings_df.empty:
            lg.warning(f"{metric_col_i_str} No pairings found.")
            continue

        for section_i, section in enumerate(d_config["sections"]):
            section_i_str = (
                f"[{section_i + 1}/{len(d_config['sections'])} {section['name']}]"
            )

            lg.info(f"{metric_col_i_str} {section_i_str} Processing section...")

            section_pairings_df = metric_pairings_df.copy()
            section_pairings_df = section_pairings_df[
                (metric_pairings_df["probability"] >= section["start"])
                & (metric_pairings_df["probability"] <= section["end"])
            ].reset_index(drop=True)

            if section_pairings_df.empty:
                lg.warning(
                    f"{metric_col_i_str} {section_i_str} section_pairings_df is empty."
                )
                continue

            most_perf_section_pairings_df = section_pairings_df.groupby(
                ["exp_variation"]
            ).apply(lambda x: get_most_perf_pairing_per_exp_var(x, metric))
            most_perf_section_pairings_df = (
                most_perf_section_pairings_df.drop_duplicates(
                    subset=["exp_variation", "most_perf_param_dict"]
                ).reset_index(drop=True)
            )

            most_perf_section_pairings_df["section_start"] = section["start"]
            most_perf_section_pairings_df["section_end"] = section["end"]
            most_perf_section_pairings_df = most_perf_section_pairings_df.reset_index(
                drop=True
            )
            most_perf_section_pairings_df = most_perf_section_pairings_df[
                [
                    "exp_variation",
                    "section_start",
                    "section_end",
                    "most_perf_param_dict",
                    "metric",
                    "possible_values",
                ]
            ]

            if (
                most_perf_section_pairings_df is None
                or most_perf_section_pairings_df.empty
            ):
                lg.warning(
                    f"{metric_col_i_str} {section_i_str} No most performant pairings found."
                )
                continue

            most_perf_pairings_df = pd.concat(
                [most_perf_pairings_df, most_perf_section_pairings_df],
                ignore_index=True,
            )

    return most_perf_pairings_df


def get_section_from_user(most_perf_pairings_df: pd.DataFrame = pd.DataFrame()):
    if most_perf_pairings_df is None or most_perf_pairings_df.empty:
        raise ValueError("No most performant pairings found.")

    sections_df = most_perf_pairings_df.copy()
    sections_df = (
        sections_df[["section_start", "section_end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    sections_dict_list = sections_df.to_dict(orient="records")

    for sec_i, section_dict in enumerate(sections_dict_list):
        click.echo(
            "{}. {} - {}".format(
                sec_i + 1, section_dict["section_start"], section_dict["section_end"]
            )
        )

    sec_i = click.prompt(
        "Which section?", type=click.IntRange(1, len(sections_dict_list))
    )

    section = sections_dict_list[sec_i - 1]

    return section["section_start"], section["section_end"]


def get_metric_from_user(most_perf_pairings_df: pd.DataFrame = pd.DataFrame()):
    if most_perf_pairings_df is None or most_perf_pairings_df.empty:
        raise ValueError("No most performant pairings found.")

    metrics = most_perf_pairings_df["metric"].unique()

    for metric_i, metric in enumerate(metrics):
        click.echo(f"{metric_i + 1}. {metric}")

    metric_i = click.prompt("Which metric?", type=click.IntRange(1, len(metrics)))

    return metrics[metric_i - 1]


def get_param_dict_from_exp_var(exp_variation_str: str = ""):
    if not exp_variation_str:
        raise ValueError(
            "No experiment variation string passed to get_param_dict_from_exp_var()."
        )

    if exp_variation_str == "":
        raise ValueError("Experiment variation string is empty.")

    if "-" not in exp_variation_str:
        raise ValueError(
            f"{exp_variation_str} is not a valid experiment variation string."
        )

    str_sections = exp_variation_str.split(" ")

    param_dict = {}
    for section in str_sections:
        key, value = section.split("-")
        param_dict[key] = value

    return param_dict


def plot_param_value_distribution_per_classification(
    variations_df: pd.DataFrame = pd.DataFrame(),
    params: list = [],
    varied_param_list: list = [],
    possible_vals_dirname_str: str = "",
):
    if variations_df is None or variations_df.empty:
        raise ValueError("No variations df passed.")

    if not params:
        raise ValueError("No parameters passed.")

    if len(params) == 0:
        raise ValueError("No parameters passed.")

    param_df = variations_df.copy()
    param_df = param_df[["most_perf_param_dict", *params]]
    param_df = param_df.drop_duplicates().reset_index(drop=True)
    param_df["most_perf_param_dict"] = param_df["most_perf_param_dict"].astype(str)

    sample_count = len(param_df)

    # Rename most_perf_param_dict to 'Classification'
    new_col_name = f"Classification (n={sample_count:,.0f})"
    param_df.rename(columns={"most_perf_param_dict": new_col_name}, inplace=True)

    fig, axs = plt.subplots(ncols=len(params), nrows=1, figsize=(5 * len(params), 5))

    for param_i, param in enumerate(params):
        ax = axs[param_i]
        sns.histplot(
            data=param_df,
            x=param,
            hue=new_col_name,
            multiple="stack",
            ax=ax,
            alpha=0.2,
        )

        ax.set_xlabel(param)
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    param_str = " vs ".join(varied_param_list)
    output_dir = "./output/parameter_analysis_tool/classification_explorer"
    output_dir = f"{output_dir}/{param_str}/{possible_vals_dirname_str}"
    filepath = f"{output_dir}/param_value_distribution_per_classification"

    plt.savefig(f"{filepath}.png")
    plt.savefig(f"{filepath}.pdf")
    plt.close()

    lg.info(f"Plot saved as {filepath}")


def get_chi_square_df(
    var_df: pd.DataFrame = pd.DataFrame(), unvaried_param_list: list = []
):
    chi_square_results = {}
    for unvaried_param in unvaried_param_list:
        if unvaried_param == "most_perf_param_dict":
            continue

        contingency_table = pd.crosstab(
            var_df[unvaried_param], var_df["most_perf_param_dict"]
        )

        chi2, p, dof, _ = scistats.chi2_contingency(contingency_table)
        chi_square_results[unvaried_param] = {"chi2": chi2, "p": p, "dof": dof}

    section_start = var_df["section_start"].iloc[0]
    section_end = var_df["section_end"].iloc[0]
    section_str = f"{int(section_start * 100)} to {int(section_end * 100)}"
    results_df = pd.DataFrame(chi_square_results).T
    results_df = results_df.rename_axis("Parameter").reset_index()
    results_df.set_index("Parameter", inplace=True)
    results_df.rename(
        columns={
            "chi2": f"{section_str} Chi-Square",
            "p": f"{section_str} P-Value",
            "dof": f"{section_str} Degrees of Freedom",
        },
        inplace=True,
    )

    return results_df


def do_chi_squared_test(
    variations_df: pd.DataFrame = pd.DataFrame(),
    unvaried_param_list: list = [],
    param_str: str = "All",
    possible_values: list = [],
    possible_vals_dirname_str: str = "",
):
    chi_square_results = {}
    chi_square_values = {}
    for unvaried_param in unvaried_param_list:
        if unvaried_param == "most_perf_param_dict":
            continue

        contingency_table = pd.crosstab(
            variations_df[unvaried_param], variations_df["most_perf_param_dict"]
        )

        chi2, p, dof, _ = scistats.chi2_contingency(contingency_table)
        chi_square_values[unvaried_param] = chi2
        chi_square_results[unvaried_param] = {"chi2": chi2, "p": p, "dof": dof}

    chi_square_df = pd.DataFrame(chi_square_values, index=["Chi-Square"]).T

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        chi_square_df,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
    )

    plt.title("Chi-Squared Test Results")
    plt.tight_layout()

    output_dir = "./output/parameter_analysis_tool/classification_explorer"
    output_dir = f"{output_dir}/{param_str}/{possible_vals_dirname_str}"
    filepath = f"{output_dir}/chi_squared_results"

    plt.savefig(f"{filepath}.png")
    plt.savefig(f"{filepath}.pdf")
    plt.close()

    lg.info(f"Plot saved as {filepath}")

    return chi_square_results


def cramers_v(chi2, n, k, r):
    return np.sqrt(chi2 / (n * min(k - 1, r - 1)))


def get_cramers_v_results_df(
    section_df: pd.DataFrame = pd.DataFrame(),
    cramers_v_results_df: pd.DataFrame = pd.DataFrame(),
):
    if section_df.empty:
        raise ValueError("Dataframe is empty.")

    unwanted_cols = [
        "possible_values",
        "exp_variation",
        "section_start",
        "section_end",
        "metric",
        "most_perf_param_dict",
    ]

    unvaried_param_list = [
        col for col in section_df.columns if col not in unwanted_cols
    ]
    for param in unvaried_param_list:
        if param not in section_df.columns:
            raise ValueError(f"Parameter {param} not in section dataframe.")

        section_df[param] = section_df[param].astype(str)

    cramers_v_df = get_cramers_v_df(section_df, unvaried_param_list)

    cramers_v_results_df = pd.concat(
        [cramers_v_results_df, cramers_v_df], ignore_index=True, axis=1
    )

    return cramers_v_results_df


def get_cramers_v_df(variations_df: pd.DataFrame = pd.DataFrame(), params: list = []):
    cramers_v_vals = {}
    for param in params:
        if param == "most_perf_param_dict":
            continue

        contingency_table = pd.crosstab(
            variations_df[param], variations_df["most_perf_param_dict"]
        )

        chi2, p, dof, ex = scistats.chi2_contingency(contingency_table)

        n = variations_df.shape[0]
        k = contingency_table.shape[0]
        r = contingency_table.shape[1]

        try:
            cramers_v_val = cramers_v(chi2, n, k, r)
        except ZeroDivisionError:
            cramers_v_val = 0

        cramers_v_vals[param] = cramers_v_val

    cramers_df = pd.DataFrame(cramers_v_vals, index=["Cramers V"]).T

    return cramers_df


def do_cramers_v_test(
    variations_df: pd.DataFrame = pd.DataFrame(),
    params: list = [],
    param_str: str = "All",
    possible_values: list = [],
    possible_vals_dirname_str: str = "",
):
    cramers_v_vals = {}
    for param in params:
        if param == "most_perf_param_dict":
            continue

        contingency_table = pd.crosstab(
            variations_df[param], variations_df["most_perf_param_dict"]
        )

        chi2, p, dof, ex = scistats.chi2_contingency(contingency_table)

        n = variations_df.shape[0]
        k = contingency_table.shape[0]
        r = contingency_table.shape[1]

        try:
            cramers_v_val = cramers_v(chi2, n, k, r)
        except ZeroDivisionError:
            cramers_v_val = 0

        cramers_v_vals[param] = cramers_v_val

    cramers_df = pd.DataFrame(cramers_v_vals, index=["Cramers V"]).T

    try:
        plot_cramers_v_values(
            cramers_df, param_str, possible_values, possible_vals_dirname_str
        )

    except ValueError as e:
        lg.error(f"Could not plot Cramer's V values:\n\t{e}")

    return cramers_v_vals


def plot_cramers_v_values(
    cramers_df: pd.DataFrame = pd.DataFrame(),
    param_str: str = "All",
    possible_values: list = [],
    possible_vals_dirname_str: str = "",
):
    sns.heatmap(
        cramers_df,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )

    plt.title("Cramer's V Values")
    plt.tight_layout()

    output_dir = "./output/parameter_analysis_tool/classification_explorer"
    output_dir = f"{output_dir}/{param_str}"
    filepath = f"{output_dir}/{possible_vals_dirname_str}/cramers_v_values"

    plt.savefig(f"{filepath}.png")
    plt.savefig(f"{filepath}.pdf")
    plt.close()

    lg.info(f"Plot saved as {filepath}")


def add_total_row_to_df(df: pd.DataFrame = pd.DataFrame()):
    total_row = df.sum(axis=0)
    total_row.name = "Total"
    df = pd.concat([df, total_row.to_frame().T])

    return df


def get_classification_df(
    pairings_df: pd.DataFrame = pd.DataFrame(),
    section_start: float = 0,
    section_end: float = 1,
    classification_df: pd.DataFrame = pd.DataFrame(),
):
    count_colname = "{} to {} Count".format(
        int(section_start * 100), int(section_end * 100)
    )

    results = pairings_df["most_perf_param_dict"]
    results = results.value_counts().to_dict()

    # Replace {'error': 'Uncategorised'} with 'Uncategorised'
    for key in list(results.keys()):
        if key == "{'error': 'Uncategorised'}":
            results["Uncategorised"] = results.pop(key)
            key = "Uncategorised"

    results_df = pd.DataFrame(results, index=[0]).T
    results_df = results_df.rename_axis("Value").reset_index()
    results_df = results_df.rename(columns={0: count_colname})
    results_df["Value"] = results_df["Value"].apply(
        lambda x: format_param_dict_to_title(x) if "{" in x else x
    )
    results_df.set_index("Value", inplace=True)

    classification_df = pd.concat(
        [classification_df, results_df],
        ignore_index=True,
        axis=1,
    )

    return classification_df


def do_correlation_analysis(
    variations_df: pd.DataFrame = pd.DataFrame(),
    unvaried_param_list: list = [],
    varied_param_list: list = [],
    possible_values: list = [],
    possible_vals_dirname_str: str = "",
):
    """
    - Turn parameter values into categories.
    - Do chi-squared test for param vs classification
    - Do cramer-v test for param vs classification
    - Do decision tree for predictive analysis
    - Plot cramers-v values as heatmap
    - Plot decision tree as tree
    """

    param_str = " vs ".join(varied_param_list) if len(varied_param_list) > 0 else "All"
    chi_square_results = do_chi_squared_test(
        variations_df,
        unvaried_param_list,
        param_str,
        possible_values,
        possible_vals_dirname_str,
    )
    cramers_v_results = do_cramers_v_test(
        variations_df,
        unvaried_param_list,
        param_str,
        possible_values,
        possible_vals_dirname_str,
    )

    correlation_table = Table(
        title="Correlation Analysis {}".format(
            [format_param_dict_to_title(val) for val in possible_values]
        ),
        show_lines=True,
    )
    columns = [
        "Parameter",
        "Chi-Squared",
        "Chi-Squared p-value",
        "DoF",
        "Cramer's V",
    ]

    headers = []
    for column in columns:
        correlation_table.add_column(column, justify="center")
        headers.append(column)

    rows = []
    for param in unvaried_param_list:
        row = [
            param,
            round(chi_square_results[param]["chi2"], 2),
            round(chi_square_results[param]["p"], 3),
            int(chi_square_results[param]["dof"]),
            round(cramers_v_results[param], 2),
        ]

        row = [str(_) for _ in row]

        # If p-val is less than 0.05, make it green
        if chi_square_results[param]["p"] < 0.05:
            row[2] = f"[green]{row[2]}[/green]"

        # If cramers v is greater than 0.5, make it green
        if cramers_v_results[param] > 0.5:
            row[4] = f"[green]{row[4]}[/green]"

        rows.append(row)
        correlation_table.add_row(*row)

    console.print(correlation_table)

    md_table = create_markdown_table(headers, rows)

    if varied_param_list == []:
        param_str = "All"
    else:
        param_str = " vs ".join(varied_param_list)

    ce_dir = "./output/parameter_analysis_tool/classification_explorer"
    md_filepath = (
        f"{ce_dir}/{param_str}/{possible_vals_dirname_str}/correlation_analysis.md"
    )

    with open(md_filepath, "w") as f:
        f.write(md_table)

    lg.info(
        f"Correlation analysis saved to {param_str}/{possible_vals_dirname_str}/correlation_analysis.md"
    )


def explore_classifications(
    most_perf_pairings_df: pd.DataFrame = pd.DataFrame(),
    varied_param_list: list = [],
    d_config: dict = {},
):
    """
    - Get section from user
    - Filter most_perf_pairings_df by section
    - Get variations
    - Get parameter values for each variation
    - Plot parameter value distribution per classification
    """

    metric = get_metric_from_user(most_perf_pairings_df)

    section_start, section_end = get_section_from_user(most_perf_pairings_df)

    unique_possible_values = get_unique_possible_values(most_perf_pairings_df)

    for possible_values in unique_possible_values:
        possible_vals_dirname_str = [
            format_param_dict_to_title(val) for val in possible_values
        ]
        possible_vals_dirname_str = "_vs_".join(possible_vals_dirname_str)
        possible_vals_dirname_str = possible_vals_dirname_str.replace("=", "-")

        os.makedirs(
            "./output/parameter_analysis_tool/classification_explorer/{}/{}".format(
                " vs ".join(varied_param_list), possible_vals_dirname_str
            ),
            exist_ok=True,
        )

        possible_values_df = most_perf_pairings_df.copy()
        possible_values_df = possible_values_df[
            possible_values_df["possible_values"] == str(possible_values)
        ].reset_index(drop=True)

        section_most_perf_pairings_df = possible_values_df.copy()
        section_most_perf_pairings_df = section_most_perf_pairings_df[
            (section_most_perf_pairings_df["section_start"] == section_start)
            & (section_most_perf_pairings_df["section_end"] == section_end)
            & (section_most_perf_pairings_df["metric"] == metric)
        ].reset_index(drop=True)
        if section_most_perf_pairings_df.empty:
            raise ValueError("No pairings found for selected section.")

        # clear_screen()
        console.print(Markdown(f"# {metric.upper()} Classification Explorer"))

        print_most_perf_pairings(
            section_most_perf_pairings_df,
            varied_param_list,
            "{}_to_{}".format(
                int(section_start * 100),
                int(section_end * 100),
            ),
            possible_values,
            d_config,
        )

        variations_df = section_most_perf_pairings_df.copy()
        variations_df = variations_df.drop_duplicates(
            subset=["exp_variation", "most_perf_param_dict"]
        ).reset_index(drop=True)

        params = []
        for var_i, var_df_row in variations_df.iterrows():
            param_dict = ast.literal_eval(var_df_row["exp_variation"])

            for key, value in param_dict.items():
                if isinstance(value, str) and "error" in value:
                    value_str = "Uncategorised"
                else:
                    value_str = value

                variations_df.loc[var_i, key] = value_str

                if key not in params:
                    params.append(key)

            if "error" in var_df_row["most_perf_param_dict"]:
                variations_df.loc[var_i, "most_perf_param_dict"] = "Uncategorised"
            else:
                variations_df.loc[var_i, "most_perf_param_dict"] = format_dict(
                    var_df_row["most_perf_param_dict"]
                )

        int_params = ["pub_count", "sub_count"]

        for param in params:
            if param in int_params:
                variations_df[param] = variations_df[param].astype(int)

            else:
                variations_df[param] = variations_df[param].astype(str)

        plot_param_value_distribution_per_classification(
            variations_df, params, varied_param_list, possible_vals_dirname_str
        )

        do_correlation_analysis(
            variations_df,
            params,
            varied_param_list,
            possible_values,
            possible_vals_dirname_str,
        )


def plot_input_distribution(df: pd.DataFrame = pd.DataFrame()):
    # Plots the distribution of input parameters

    if df is None or df.empty:
        raise ValueError("No dataframe passed to plot_input_distribution().")
    if not PARAM_COLS:
        raise ValueError("No parameter columns defined in PARAM_COLS.")

    for col in PARAM_COLS:
        if col not in df.columns:
            raise ValueError(f"Parameter column {col} not found in dataframe.")

    lg.info("Plotting input parameter distribution...")

    df_params = df[PARAM_COLS].copy()
    df_params = df_params.drop_duplicates().reset_index(drop=True)
    df_params["datalen_bytes"] = df_params["datalen_bytes"].astype(int)
    df_params["pub_count"] = df_params["pub_count"].astype(int)
    df_params["sub_count"] = df_params["sub_count"].astype(int)
    df_params["durability"] = df_params["durability"].astype(str)

    df_params = df_params.rename(
        columns={
            "datalen_bytes": "Data Length (B)",
            "pub_count": "Publisher Count",
            "sub_count": "Subscriber Count",
            "use_reliable": "Reliability",
            "use_multicast": "Multicast",
            "durability": "Durability",
        }
    )

    num_params = len(df_params.columns)  # Number of parameters
    # Create 2 chunks of 3 parameters each
    lls_chunks = [df_params.columns[i : i + 3] for i in range(0, num_params, 3)]
    for i_chunk, ls_chunk in enumerate(lls_chunks):
        df_chunk = df_params[list(ls_chunk)].copy()
        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(8, 12))

        for i, col in enumerate(df_chunk.columns):
            ax = axs[i]
            if df_chunk[col].dtype == "int64" or df_chunk[col].dtype == "float64":
                sns.histplot(df_chunk[col], kde=False, ax=ax, color="white", bins=30)

            else:
                sns.countplot(
                    y=df_chunk[col], ax=ax, edgecolor="black", facecolor="white"
                )
                for p in ax.patches:
                    width = p.get_width()
                    ax.text(
                        width + 0.1,
                        p.get_y() + p.get_height() / 2,
                        f"{int(width)}",
                        ha="left",
                        va="center",
                    )

            ax.set_title(f"Distribution of {col}")
            ax.grid(axis="x", linestyle="--", alpha=0.2)
            ax.grid(axis="y", linestyle="--", alpha=0.2)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        output_path = f"./output/input_parameter_distribution_{i_chunk + 1}"
        plt.savefig(f"{output_path}.png")
        plt.savefig(f"{output_path}.pdf")
        plt.close()

        lg.info(f"Input parameter distribution plot saved to {output_path}.")


if __name__ == "__main__":
    with timer.Timer():
        try:
            main(d_config)
        except Exception as e:
            Notifier.notify(f"Error: {str(e)}", title="Parameter Analysis Tool")
            lg.exception(e)
