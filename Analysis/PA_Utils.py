# Import libraries
import glob
from tqdm import tqdm
import datetime
import re
import pandas as pd
import warnings
import os
import pickle
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from scipy.optimize import linear_sum_assignment
from matplotlib.backends.backend_pdf import PdfPages


def get_animal_data(data_path, Animal_ID, protocol, data_type):
    """
    Args:
        data_path (str): path to the data folder
        Animal_ID (str): animal ID
        protocol (str): protocol name
        data_type (str): data type (e.g. 'Trial_Summary', 'Detected_Licks')
    Returns:
        animal_df (pd.DataFrame): dataframe containing all the data for the animal
    """

    subject_folders = glob.glob(data_path + Animal_ID +'/*') 
    protocol_folders = list(filter(lambda x: protocol + '_' + Animal_ID in x, subject_folders)) 

    if len(protocol_folders) == 0:
        warnings.warn(f"{Animal_ID}: all_folders empty")
        return 0

    sessions_data = []
    for folder in tqdm(protocol_folders, position=0, leave=True, desc = 'Processing ' + Animal_ID ):  
        folder_files = glob.glob(folder +'/**/'+data_type +'*.csv', recursive = True)
        for file in folder_files:
            try:
                session_df = pd.read_csv(file)
                date_pattern = r'\d{4}_\d{1,2}_\d{1,2}'
                date = re.search(date_pattern, file).group(0)
                date_obj = datetime.datetime.strptime(date, '%Y_%m_%d')
                formatted_date = date_obj.strftime('%Y/%m/%d')
                session_df.insert(0, 'Date', formatted_date)
                session_df['File_ID'] = file  # Add file identifier
                sessions_data.append(session_df)
            except pd.errors.EmptyDataError:
                pass

        animal_df = pd.concat(sessions_data, axis=0, ignore_index=True)
        # Create a list of columns for sorting
        sort_columns = ['Date']

        # Check which of the columns ('Trial_End_Time' or 'Time') exists in the DataFrame
        if 'Trial_End_Time' in animal_df.columns:
            sort_columns.append('Trial_End_Time')
        elif 'Time' in animal_df.columns:
            sort_columns.append('Time')

        # Sort the DataFrame by the determined columns
        animal_df = animal_df.sort_values(by=sort_columns)

        date_list = []
        for date in animal_df.Date.unique():
            date_df = animal_df[animal_df.Date == date].reset_index(drop=True)
            date_list.append(date_df)
            
        animal_df = pd.concat(date_list, axis=0, ignore_index=True)
        # animal_df['block'] =  (animal_df['TrialNumber'] == 1).cumsum() # to be consistent with ELV, can have several blocks per date
        # animal_df['Row_Number'] = animal_df.groupby('block').cumcount() + 1 # get row number for every block
        # animal_df.set_index(['block', 'Row_Number'], inplace=True) # order df by block and row number
        # animal_df.reset_index(inplace=True)
        # animal_df = animal_df.drop('TrialNumber', axis=1)
        # animal_df.rename(columns={'Row_Number': 'Trial_Number'}, inplace=True)

    return animal_df

def process_raw_data(df):
    '''
    Args:
        df: pandas dataframe of raw data
    Returns:
        df: pandas dataframe of processed data
    '''
    df = df.copy()  # Create a copy

    df['Correct'] = np.where(df['Trial_Outcome']=='Correct', 1, 0)
    df['Incorrect'] = np.where(df['Trial_Outcome']=='Incorrect', 1, 0)
    # df['No_Response'] = np.where(df['Trial_Outcome']=='Abort', 1, 0)
    df['choice'] = np.where(df['First_Lick']=='Right', 1, 0) # 

    # # Check if 'Trial_Outcome' is not NaN and not empty
    # if not df['Trial_Outcome'].isnull().all() and (df['Trial_Outcome'] != '').any():
    #     # Create the 'Correct' column
    #     df['Correct'] = np.where(df['Trial_Outcome'] == 'Correct', 1, 0)

    #     # Create the 'Incorrect' column
    #     df['Incorrect'] = np.where(df['Trial_Outcome'] == 'Incorrect', 1, 0)

    #     # Create the 'choice' column
    #     df['choice'] = np.where(df['First_Lick'] == 'Right', 1, 0)
    
    df = df.rename(columns={'Animal_ID': 'Participant_ID', 
                                        'Trial_Number': 'Trial', 
                                        'Correct': 'correct', 
                                        'Abort_Trial': 'No_response'})

    df['Rule_Right'] = (
    ((df['Air_Puff_Contingency'] == 'Pro') & (df['Air_Puff_Side'] == 'Right')) | 
    ((df['Air_Puff_Contingency'] == 'Anti') & (df['Air_Puff_Side'] == 'Left'))
    ).astype(int)
    df['Choice_Rule_Diff'] = df['choice'] - df['Rule_Right']

    # Define the bins
    bins_negative = [-1.0, -0.75, -0.5, -0.25, 0.0]
    bins_positive = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Create masks for negative and positive values
    mask_negative = df['Stim_Relative'] < 0
    mask_positive = df['Stim_Relative'] > 0

    # Bin the negative values with left inclusivity
    binned_negative = pd.cut(df[mask_negative]['Stim_Relative'], bins=bins_negative, right=False, labels = np.arange(-0.875, 0, 0.25))

    # Bin the positive values with right inclusivity
    binned_positive = pd.cut(df[mask_positive]['Stim_Relative'], bins=bins_positive, right=True, labels = np.arange(0.125, 1, 0.25))

    # Convert the binned categories to strings
    binned_negative = binned_negative.astype(str)
    binned_positive = binned_positive.astype(str)

    # Assign binned values using masks
    df.loc[mask_negative, 'stim_relative_binned'] = binned_negative
    df.loc[mask_positive, 'stim_relative_binned'] = binned_positive
    # df['stim_relative_binned'] = df['stim_relative_binned'].astype('category')
    df['stim_relative_binned'] = pd.to_numeric(df['stim_relative_binned'], errors='coerce')

    
    return df

def compute_air_puff_side(row):
    if pd.isna(row['Air_Puff_Side']):
        if row['Air_Puff_Contingency'] == 'Pro' and row['Correct'] == True:
            return row['First_Lick']
        elif row['Air_Puff_Contingency'] == 'Pro' and row['Correct'] == False:
            return 'Right' if row['First_Lick'] == 'Left' else 'Left'
        elif row['Air_Puff_Contingency'] == 'Anti' and row['Correct'] == True:
            return 'Right' if row['First_Lick'] == 'Left' else 'Left'
        elif row['Air_Puff_Contingency'] == 'Anti' and row['Correct'] == False:
            return row['First_Lick']
    else:
        return row['Air_Puff_Side']

def grab_columns(df, columns):
    return df[columns]

def add_block_column(df, participant_col, datetime_col, trial_end_time, trial_col):
    # Ensure dataframe is sorted by participant_col and trial_col
    df = df.sort_values(by=[participant_col, datetime_col, trial_end_time])

    # Create a temporary 'block_start' column which is 1 when trial_col is 1 and 0 otherwise
    df['block_start'] = (df[trial_col] == 1).astype(int)

    # For each participant_col, compute the cumulative sum of 'block_start' to get the block number
    df['block'] = df.groupby(participant_col)['block_start'].cumsum()

    # Drop the temporary 'block_start' column
    df = df.drop(columns='block_start')

    return df

def plot_performance_and_bias_all(df, participant_ids, color_dict, save_path, show_plot=False):
    # add docstring
    '''
    Plot performance and bias for all participants in a single figure.
    Args:
        df (DataFrame): DataFrame with the data
        participant_ids (list): list of participant IDs
        color_dict (dict): dictionary with the colors for the different conditions
        save_path (str): path to save the figure
        show_plot (bool): whether to show the plot or not
    Returns:
        Returns the plot if show_plot is True.
    '''
    
    plot_settings = [
        {
            'x': 'Date',
            'y': 'correct',
            'xlabel': 'Date',
            'ylabel': 'Proportion Correct',
            'hue': 'Air_Puff_Contingency',
            'sort_values': 'Datetime',
            'legend_title': 'Air Puff Contingency',
            'ylim': [0, 1],
            'hline': 0.5,
            'title': 'Performance'
        },
        {
            'x': 'Date',
            'y': 'Choice_Rule_Diff',
            'xlabel': 'Date',
            'ylabel': 'Bias',
            'hue': 'Air_Puff_Contingency',
            'sort_values': 'Datetime',
            'legend_title': 'Air Puff Contingency',
            'ylim': [-1, 1],
            'hline': 0,
            'title': 'Bias'
        },
        {
            'x': 'stim_relative_binned',
            'y': 'choice',
            'xlabel': 'Stimulus Relative',
            'ylabel': 'Proportion Chose Right',
            'hue': 'Air_Puff_Side',
            'sort_values': 'stim_relative_binned',
            'legend_title': 'Air Puff Side',
            'ylim': [0, 1],
            'hline': 0.5,
            'title': 'Psychometric'
        }
    ]

    # Open the PDF file
    with PdfPages(save_path) as pdf:
        for participant_id in participant_ids:
            participant_df = df[df['Participant_ID'] == participant_id]
            participant_df = participant_df[participant_df['No_response'] == False]

            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            for ax, settings in zip(axes, plot_settings):
                participant_df = participant_df.sort_values(by=settings['sort_values'])
                sns.pointplot(x=settings['x'], y=settings['y'], hue = settings['hue'], data=participant_df, palette=color_dict, ax=ax, ci=95)
                ax.set_xlabel(settings['xlabel'])
                ax.set_ylabel(settings['ylabel'])
                ax.set_title(settings['title'])
                ax.set_ylim(settings['ylim'])
                ax.legend(loc='lower right', title=settings['legend_title'])
                ax.axhline(y=settings['hline'], color='k', linestyle='--')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                # Check the number of ticks on the x-axis
                if len(ax.get_xticklabels()) > 10:
                    for i, label in enumerate(ax.get_xticklabels()):
                        if i % 5 != 0:
                            label.set_visible(False)

            # Add a big title for both subplots
            fig.suptitle(f'Participant ID: {participant_id}', fontsize=16, y=0.99)

            plt.tight_layout()
            
            # Save the current figure to the PDF
            pdf.savefig(fig)

            if show_plot:
                plt.show()
            else:
                plt.close()

def plot_within_session_perf(df, Participant_ID, save_path, show_plot=False, jitter=0.05, background_shading=True):
    with PdfPages(save_path) as pdf:
        participant_df = df[df['Participant_ID'] == Participant_ID]
        participant_df = participant_df[participant_df['No_response'] == False]

        for block in participant_df.block.unique():
            block_df = participant_df[participant_df['block'] == block].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(18, 6))
            y_labels = ['Pro Go L', 'Go L', 'Anti Go L', 'Pro Go R', 'Go R', 'Anti Go R']
            ax.set_yticks([0, 0.5, 1, 2, 2.5, 3])
            ax.set_yticklabels(y_labels)
            ax.set_xlabel('(Valid) Trial')
            ax.set_ylabel('Condition')

            # Background shading along x-axis
            if background_shading:
                start_index = 0
                current_type = block_df['Air_Puff_Contingency'].iloc[0]
                for i, row in block_df.iterrows():
                    if row['Air_Puff_Contingency'] != current_type:
                        color = 'lightgreen' if current_type == 'Pro' else 'lightcoral'
                        ax.axvspan(start_index - 0.5, i - 0.5, facecolor=color, alpha=0.2)
                        start_index = i
                        current_type = row['Air_Puff_Contingency']
                # Shade the final block
                color = 'lightgreen' if current_type == 'Pro' else 'lightcoral'
                ax.axvspan(start_index - 0.5, i + 0.5, facecolor=color, alpha=0.2)


            # Mapping conditions to y-values using a dictionary
            y_map = {
                ('Pro', 0): 0,
                ('Anti', 0): 1,
                ('Pro', 1): 2,
                ('Anti', 1): 3
            }

            colors = ['green' if correct else 'red' for correct in block_df['correct']]
            y_values = block_df.apply(lambda row: y_map.get((row['Air_Puff_Contingency'], row['choice'])), axis=1)

            # Add jitter to y-values
            y_values += np.random.uniform(-jitter, jitter, len(y_values))

            # Plotting the points
            ax.scatter(block_df.index, y_values, color=colors, s=50, alpha=0.5)

            # Plotting black dots based on choice
            for index, choice in block_df['choice'].items():
                y = 0.5 if choice == 0 else 2.5
                y += np.random.uniform(-jitter, jitter)
                ax.plot(index, y, 'o', color='black', markersize=7, alpha=0.5)

            plt.xlim(-3, block_df.index[-1] + 3)
            plt.title(f'Participant {Participant_ID} on {block_df.Date[0]} (Block {block})')
            plt.tight_layout()

            # Save figure to pdf
            pdf.savefig(fig)

            # Show figure
            if show_plot:
                plt.show()
            else:
                plt.close()

                