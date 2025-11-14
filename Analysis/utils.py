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


# Define functions

def get_animal_data(data_path, Animal_ID, protocol, data_type):
    '''generate pandas dataframe with all data from one animal on a given protocol/family of protocols

        Args:
            Animal_ID (str): mouse name
            protocol (str): name of protocol
            file_type (str): fyle type
            data_path (str): data path
        Returns:
            (pd.dataframe): concatenated dataframe of all the data for the given file type and animal
    '''

    subject_folders = glob.glob(data_path + Animal_ID +'/*') 

    protocol_folders = list(filter(lambda x: protocol in x, subject_folders)) 
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
                sessions_data.append(session_df)
            except pd.errors.EmptyDataError:
                pass

        animal_df = pd.concat(sessions_data, axis=0, ignore_index=True)
        # animal_df = animal_df.sort_values(by="Date")  #order data by date  
        animal_df = animal_df.sort_values(['Date', 'Time'])
        date_list = []
        for date in animal_df.Date.unique():
            date_df = animal_df[animal_df.Date == date].reset_index(drop=True)
            date_list.append(date_df)
            
        animal_df = pd.concat(date_list, axis=0, ignore_index=True)

        animal_df['block'] =  (animal_df['TrialNumber'] == 1).cumsum() # to be consistent with ELV, can have several blocks per date
        animal_df['Row_Number'] = animal_df.groupby('block').cumcount() + 1 # get row number for every block
        animal_df.set_index(['block', 'Row_Number'], inplace=True) # order df by block and row number
        animal_df.reset_index(inplace=True)
        animal_df = animal_df.drop('TrialNumber', axis=1)
        animal_df.rename(columns={'Row_Number': 'Trial_Number'}, inplace=True)
        animal_df['Correct'] = np.where(animal_df['TrialOutcome']==1, 1, 0)
        animal_df['Incorrect'] = np.where(animal_df['TrialOutcome']==0, 1, 0)
        animal_df['No_Response'] = np.where(animal_df['TrialOutcome']==-1, 1, 0)
        animal_df['Choice'] = np.where(animal_df['FirstLick']==2, 2, 1) # choice is 2 for right lick, 1 for left lick
        
    return animal_df

def psychometric (s, mu , sigma , lapse1 , lapse2):
    """
    Psychometric function to be used for fitting purposes
    :param s: stimulus (distance from the boundary)
    :param m: mean of the gaussian distribution
    :param sigma: std of the gaussian distribution (related to the slope of the curve)
    :param lapse1: lower lapse of the curve (or lambda)
    :param lapse2: upper lapse of the curve (or lambda)
    :return: probability of choosing option B (going left)
    """
    psyc = lapse1 + (1 - lapse1 - lapse2) * norm.cdf(s, mu, sigma)

    return psyc

def psychometric_derivative(s, mu, sigma, lapse1, lapse2):
    z = (s - mu) / sigma
    d = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) / sigma
    return (1 - lapse1 - lapse2) * d

def bootstrap(samples, choices, num_iters = 500): 
    x_store =  np.zeros((num_iters, len(x)))
    y_store =  np.zeros((num_iters, len(x)))

    for iters in range(num_iters):
        boot = np.random.choice(np.arange(len(x)),len(x))
        x_boot = samples[boot]
        y_boot = choices[boot]
        x_store[iters,:] = x_boot
        y_store[iters,:] = y_boot
    return(x_store, y_store)

def normalize_amp(val, min_amp=60, max_amp=92):
    # Check if val is within the min_amp and max_amp range
    if not np.isnan(val) and min_amp <= val <= max_amp:
        # Normalize val to the range [-1, 1]
        return 2 * (val - min_amp) / (max_amp - min_amp) - 1
    else:
        return val
        
def normalize_oct(val):
    if not np.isnan(val) and 0 <= val <= 2:
        return (val - 1) / 1
    return val

def filter_min_trials(df, threshold):
    # Create an empty list to store the filtered dataframes
    filtered_dfs = []

    # Iterate over unique animal IDs
    for animal_id in df['Participant_ID'].unique():
        # Create a dataframe for the current animal ID
        animal_df = df[df['Participant_ID'] == animal_id]

        # Iterate over unique sessions for the current animal ID
        for session in animal_df['block'].unique():
            # Create a dataframe for the current session
            session_df = animal_df[animal_df['block'] == session]

            # Count the number of rows where 'Abort_Trial' == False
            count = session_df.query("No_response == False").shape[0]

            # If the count is greater than or equal to the threshold,
            # append the session dataframe to the filtered_dfs list
            if count >= threshold:
                filtered_dfs.append(session_df)

    # Concatenate all filtered dataframes into a single dataframe
    filtered_df = pd.concat(filtered_dfs, ignore_index=True)

    return filtered_df

def make_blocks_equal(df):
    """Takes dataframe and ensures that all participants and distributions have the same number of blocks
    Args:
        df(pd.dataframe): dataframe to be processed
    Returns: 
        dataframe with sessions exceeding lowest common session for each participant and distribution removed
    """
    block_counts = df.groupby(['Participant_ID','Distribution'])['block'].nunique()
    block_counts = block_counts.reset_index()
    min_dist_dict = []
    for distribution in block_counts.Distribution.unique():
        distrbution_block_counts = block_counts[block_counts.Distribution == distribution]
        distribution_df = df[df.Distribution == distribution]
        dist_block_min = distrbution_block_counts.block.min()
        min_dist_dict.append(distribution_df[distribution_df.dist_block <= dist_block_min])
    filtered_df = pd.concat(min_dist_dict)
    return filtered_df

def check_distributions(df, n=3):
    """Takes a dataframe and ensures at least n distributions are present.

    Args:
        df (pd.DataFrame): The dataframe to be processed.
        n (int): The minimum number of distributions to ensure are present.

    Returns:
        A filtered dataframe with participants without at least n distributions removed and reindexed rows.
    """
    filtered_df = df.groupby('Participant_ID').filter(lambda x: x['Distribution'].nunique() >= n)
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df
    
def set_plot_font(font_family='sans-serif',
                  title_font_size=20, label_font_size=20,
                  serif_font='Times New Roman',
                  sans_serif_font='Helvetica',
                  monospace_font='Courier New',
                  font_file=None,
                  text_font_size=20):  # Added 'text_font_size' parameter


    if font_file:
        font = fm.FontProperties(fname=font_file)
        font_family = font.get_name()

    sns.set(
        style='ticks',
        font=font_family,
        font_scale=1,
        rc={
            'axes.labelsize': label_font_size,
            'axes.titlesize': title_font_size,
            'xtick.labelsize': label_font_size,
            'ytick.labelsize': label_font_size,
            'legend.fontsize': label_font_size,
            'font.serif': serif_font,
            'font.sans-serif': sans_serif_font,
            'font.monospace': monospace_font
        }
    )

    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.serif'] = serif_font
    plt.rcParams['font.sans-serif'] = sans_serif_font
    plt.rcParams['font.monospace'] = monospace_font
    plt.rcParams['font.size'] = text_font_size  # Set the font size globally

    plt.rc('xtick', labelsize=label_font_size)
    plt.rc('ytick', labelsize=label_font_size)
    plt.rc('axes', labelsize=label_font_size, titlesize=title_font_size)
    plt.rc('legend', fontsize=label_font_size, title_fontsize=title_font_size)

def filter_transitions_sessions(df, distributions_to_filter, session_threshold):
    """Takes dataframe removes sessions beyond a certain threshold for specified distributions
    Args:
        df(pd.dataframe): dataframe to be processed
        distributions_to_filter(list): list of distributions to filter
        session_threshold(int): session beyond which data in retained
    Returns: 
        dataframe with sessions below session_threshold for specified distributions removed 
    """ 
    my_list = []
    for participant in df.Participant_ID.unique():
        participant_df = df[df.Participant_ID == participant]
        for distribution in participant_df.Distribution.unique():
            distribution_df = participant_df[participant_df.Distribution == distribution]
            if distribution in distributions_to_filter:
                distribution_df_filtered = distribution_df[distribution_df.dist_block > session_threshold]
                my_list.append(distribution_df_filtered)
            else:
                my_list.append(distribution_df)
    filtered_df = pd.concat(my_list)
    filtered_df = filtered_df.reset_index(drop = True)
    return filtered_df

def gen_animal_list(n,m):
    animal_list = []
    for i in range(n, m+1):
        animal_name = f'QP{i:03}'
        animal_list.append(animal_name)
    return animal_list

def add_dist_block(df):
    # add block change column to detect when block changes for each participant and distribution
    df['block_change'] = df.groupby(['Participant_ID', 'Distribution'])['block'].diff().ne(0).astype(int)
    
    # add dist_block column to keep track of block changes within each participant and distribution
    df['dist_block'] = df.groupby(['Participant_ID', 'Distribution'])['block_change'].cumsum()
    
    # drop the block_change column since it's no longer needed
    df.drop('block_change', axis=1, inplace=True)
    
    return df

def convert_stim_dur_to_ms(df):
    def time_str_to_ms(time_str):
        if pd.isna(time_str):
            return time_str

        parts = time_str.split(':')
        hh, mm, ss = 0, 0, 0

        if len(parts) == 3:
            hh, mm, ss = int(parts[0]), int(parts[1]), float(parts[2])
        elif len(parts) == 2:
            hh, mm = int(parts[0]), int(parts[1])

        total_seconds = hh * 3600 + mm * 60 + ss
        total_milliseconds = int(total_seconds * 1000)

        return total_milliseconds

    df['Stim_Dur'] = df['Stim_Dur'].apply(time_str_to_ms)
    return df

def add_missing_stim_dur_data(df):
    df['Stim_Dur'] = df['Stim_Dur'].fillna(300.0)
    return df

def add_stim_relative(df, animal_ids):
    def process_animal_data(df, animal_id):
        df_filtered = df[df['Animal_ID'] == animal_id].copy()
        
        animal_num = int(animal_id[2:])
        
        if animal_num <= 71:
            df_filtered.loc[:, 'Octave'] = df_filtered.apply(lambda row: round((np.log(row.Tone/4)) / np.log(2), 7), axis=1)
            df_filtered.loc[:, 'stim_relative'] = np.where(np.isnan(df_filtered['Octave']),  df_filtered['WN_Amp'].apply(normalize_amp, min_amp=60, max_amp=92), df_filtered['Octave'].apply(normalize_oct))
        else:
            df_filtered.loc[:, 'stim_relative'] = df_filtered['WN_Amp'].apply(normalize_amp, min_amp=50, max_amp=82)

        df_filtered.loc[:, 'stim_relative_binned'], cut_bin = pd.qcut(df_filtered['stim_relative'],  q=8, labels=np.arange(-0.875, 0.9, 0.25), retbins=True)

        return df_filtered

    # Initialize an empty DataFrame to store the processed data
    processed_df = pd.DataFrame()

    # Regular expression pattern to match animal_ids in the 'QP00n' format
    qp_pattern = re.compile(r"^QP\d{3}$")

    for animal_id in animal_ids:
        if qp_pattern.match(animal_id):
            processed_animal_df = process_animal_data(df, animal_id)
            processed_df = pd.concat([processed_df, processed_animal_df], ignore_index=True)
    
    return processed_df

def relabel_ELV(df):
    # convert tone to octave
    # df['Octave'] = df.apply(lambda row: round((np.log(row.Tone/4)) / np.log(2), 7), axis=1)
    df = add_stim_relative(df, df.Animal_ID.unique())

    # rename Distribution values like Victor
    df['Distribution'] = df['Distribution'].replace({'Asym_Left': 'Asym_left', 'Asym_Right': 'Asym_right'})
    
    # label columns like ELV
    df['block'] = df['block'].apply(lambda x: x - 1) # apply 0 indexing to block
    df['Trial_Number'] = df['Trial_Number'].apply(lambda x: x - 1) # apply 0 indexing to TrialNumber
    df['Tone'] = df['Tone'].apply(lambda x: x * 1000) # go from kHz to Hz
    # df['stimulus'] = np.where(np.isnan(df['Tone']), df['WN_Amp'], df['Tone'])
    # df['stim_relative'] = np.where(np.isnan(df['Octave']), 
    #                                df['WN_Amp'].apply(normalize_amp), df['Octave'].apply(normalize_oct))
    # df['stim_relative_binned'], cut_bin = pd.qcut(df['stim_relative'],  
    #                                           q = 8, labels = np.arange(-0.875,0.9,0.25),retbins = True)
        # TODO: need to perform binning like Victor
    # df['stim_type'] = np.where(np.isnan(df['Tone']), 'WN', np.where(np.isnan(df['WN_Amp']), 'PT', 'NA'))
    
    # df = add_stim_relative(df, df.Animal_ID.unique())
    df['stim_type'] = np.where(pd.isna(df['Tone']), 'WN', np.where(pd.isna(df['WN_Amp']), 'PT', 'NA'))
    df['choice'] = df['FirstLick'].apply(lambda x: x - 1) # 0 for left (A), 1 for B (right)
    
    df = df.rename(columns={'Animal_ID': 'Participant_ID', 'Trial_Number': 'Trial', 
                            'Correct': 'correct', 'AbortTrial': 'No_response'})
    df['Rule_Right'] = np.where(df['stim_relative']>0, 1, 0)

    df['Choice_Rule_Diff'] = df['choice'] - df['Rule_Right']
    # only keep columns used by ELV
    df = df[['Participant_ID', 'Distribution', 'Trial', 'correct', 'block', 'Date',
            #  'stimulus',  
             'stim_relative', 
             'stim_relative_binned', 
             'choice', 'No_response', 
             'Response_Latency', 'stim_type', 'Choice_Rule_Diff', 'Anti_Bias', 
             'Stim_Dur', 'Time', 'GoCueDur', 'ITI', 'ReponseWindow',
             'Opto_ON', 'Light_Freq', 'Perc_Opto_Trials', 'Opto_Onset', 
             'Opto_Offset', 'LED_ON' # opto params seem to prevwent Victor's code from running
             ]]
        
    # df = add_missing_stim_dur_data(df)
    df = convert_stim_dur_to_ms(df)
    df = add_missing_stim_dur_data(df)

    # label relevant rows as np.nan for no_response trials
    df.loc[df['No_response'] == True, 'correct'] = np.nan
    df.loc[df['No_response'] == True, 'choice'] = np.nan
    df.loc[df['No_response'] == True, 'Response_Latency'] = np.nan
    df.loc[df['No_response'] == True, 'Choice_Rule_Diff'] = np.nan

    return df

def add_cohort_ids(df, cohort_lists):
    df['Cohort_ID'] = None  # Initialize Cohort_ID column with None
    
    for i, cohort in enumerate(cohort_lists, start=1):
        df.loc[df['Participant_ID'].isin(cohort), 'Cohort_ID'] = f'Cohort_{i}'
    
    return df

def threshold_cutoff(df, threshold=0.7):
    # group the dataframe by 'Participant_ID' and 'block'
    valid_blocks = df.groupby(['Participant_ID', 'block'])['correct'].mean()
    
    # filter out blocks where average 'correct' is less than threshold and add new block index
    valid_blocks = valid_blocks[valid_blocks >= threshold].reset_index()
    
    # re-index blocks
    valid_blocks['new_block_index'] = valid_blocks.groupby('Participant_ID').cumcount()
    valid_blocks = valid_blocks[['Participant_ID', 'block', 'new_block_index']]
    
    # merge with original DataFrame to filter out invalid blocks
    filtered_df = df.merge(valid_blocks, on=['Participant_ID', 'block'])
    
    # remove old block column and rename new block column
    filtered_df = filtered_df.drop(columns=['block']).rename(columns={'new_block_index': 'block'})
    
    return filtered_df

def filter_lapses(df, threshold=0.80):
    """
    Takes a pandas dataframe `df` and removes each `block` 
    for each `Participant` where the performance on lapses (`df[df.stim_relative_binned == -0.875]`
    or `df[df.stim_relative_binned == 0.875]`) is less than threshold. 
    
    Parameters:
        df (pandas.DataFrame): The dataframe containing the data to filter.
        threshold (float): threshold performance on lapse rates
        
    Returns:
        pandas.DataFrame: The filtered dataframe.
    """
    # Create a new dataframe to store the filtered data
    filtered_df = pd.DataFrame()

    # Loop through each participant
    for participant in df['Participant_ID'].unique():
        # Get the data for this participant
        participant_data = df[df['Participant_ID'] == participant]
        
        # Initialize a variable to keep track of the new block index
        new_block_index = 0

        # Loop through each block
        for block in participant_data['block'].unique():
            # Get the data for this block
            block_data = participant_data[participant_data['block'] == block]

            # Check if the performance is less than threshold for the relevant stim_relative_binned values
            if ((block_data[block_data['stim_relative_binned'] == -0.875]['correct'].mean() < threshold) or 
                (block_data[block_data['stim_relative_binned'] == 0.875]['correct'].mean() < threshold)):
                # If the performance is less than threshold, don't include this block in the filtered data
                continue

            # If the performance is greater than or equal to threshold, include this block in the filtered data
            # Add a new column with the updated block index
            block_data['block'] = new_block_index
            
            # Add the updated block data to the filtered data
            filtered_df = pd.concat([filtered_df, block_data])

            # Increment the new block index
            new_block_index += 1

    # Re-index the rows in the filtered data
    filtered_df = filtered_df.reset_index(drop=True)

    # Return the filtered data
    return filtered_df

def sort_consec_dist(df, desired_order):
    """
    Takes a dataframe and removes all blocks that do not followed the imposed order:

    Args:
        df (pd.DataFrame): original dataframe
        desired_order (str): desired distribution order e.g. ['Uniform', 'Asym_right', 'Asym_left']

    Returns:
        pd.DataFrame: A new dataframe where blocks not following the imposed order are removed
    """
    
    filtered_df = df[df['Distribution'].notna()]
    ordered_distributions = []
    concatenated_dfs = []
    for participant in filtered_df.Participant_ID.unique():
        last_dist = None  # reset last_dist for each new participant
        participant_df = filtered_df[filtered_df.Participant_ID == participant]
        block_dfs = []
        for block in participant_df.block.unique():
            block_df = participant_df[participant_df.block == block]
            if (last_dist is not None) and (desired_order.index(block_df.Distribution.unique()[0]) 
                                            < desired_order.index(last_dist)):
                continue  # discard element not in the desired order
            block_dfs.append(block_df)
            last_dist = block_df.Distribution.unique()[0]
        # Discard any trailing 'Uniform's or 'Asym_right's
        while block_dfs and block_dfs[-1]['Distribution'].iloc[0] in ['Uniform', 'Asym_right']:
            block_dfs.pop()
        concatenated_dfs.append(pd.concat(block_dfs))
    filtered_df = pd.concat(concatenated_dfs)
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def calculate_entropy(seq):
    ones_count = seq.count(1)
    zeros_count = seq.count(0)
    total_count = ones_count + zeros_count
    ones_prob = ones_count / total_count
    zeros_prob = zeros_count / total_count
    entropy = 0
    if ones_prob > 0:
        entropy -= ones_prob * math.log2(ones_prob)
    if zeros_prob > 0:
        entropy -= zeros_prob * math.log2(zeros_prob)
    return entropy

def calc_conf_interval(group):
    # calculate the mean and count of 'choice' for the group
    mean = group['choice'].mean()
    count = group['choice'].count()
    
    # calculate the standard error of the mean
    std_error = np.sqrt(mean*(1-mean)/count)
    
    # calculate the 95% confidence interval
    conf_interval = 1.96 * std_error
    
    # return the mean and confidence interval as a series
    return pd.Series({'mean': mean, 'count': count, 'conf_interval': conf_interval})

def plot_dist_hist(df, color_dict, num_bins):
    # valid_df = df[df.No_response == False]
    grouped = df.groupby('Distribution')
    figsize = (4*len(grouped), 4) 
    fig = plt.figure(figsize=figsize)  

    for i, (dist_name, dist_df) in enumerate(grouped):
        ax = fig.add_subplot(1, len(grouped), i+1)
        ax.hist(dist_df['stim_relative'], bins=num_bins, color=color_dict[dist_name])
        ax.set_title(dist_name)
        ax.set_xlabel('Distance from Boundary', color='k')
        ax.set_ylabel('Count', color='k')

    plt.tight_layout()
    plt.show()
    
def check_valid_participant_ids(df, participant_ids):
    valid_ids = df.Participant_ID.unique()
    valid_participant_ids = []
    invalid_participant_ids = []
    for pid in participant_ids:
        if pid in valid_ids:
            valid_participant_ids.append(pid)
        else:
            invalid_participant_ids.append(pid)
    if invalid_participant_ids:
        print(f"Warning: Invalid participant IDs: {invalid_participant_ids}")
    return valid_participant_ids    

def plot_performance_and_bias(df, participant_ids, distributions, 
                              background_shading = True, 
                              color_dict = {'Uniform': '#2F9A76', 'Asym_left': '#007dc7', 'Asym_right': '#c23f63'}):
    """
    Plots the performance and bias for the specified participant IDs and distributions, with the option to include background
    shading and custom colors for different distributions.

    Args:
        df (pandas.DataFrame): The data frame to use for the plot.
        participant_ids (list): A list of participant IDs to include in the plot.
        distributions (list): A list of distributions to include in the plot.
        background_shading (bool): If True, adds background shading to the plot to indicate the distribution of each block.
        color_dict (dict): A dictionary that maps each distribution to a color.

    Returns:
        None
    """
    valid_participant_ids = check_valid_participant_ids(df, participant_ids)
       
    filtered_df = df[df.No_response == False]
    filtered_df = filtered_df[filtered_df['Participant_ID'].isin(participant_ids)]
    filtered_df = filtered_df[filtered_df['Distribution'].isin(distributions)]

    max_cols = 3
    num_participants = len(valid_participant_ids)
    cols = min(num_participants, max_cols)
    rows = int(np.ceil(num_participants / cols))
    fig = plt.figure(figsize=(cols*4, rows*4))

    for i, participant_id in enumerate(valid_participant_ids):
        participant_df = filtered_df[filtered_df.Participant_ID == participant_id]
        ax1 = plt.subplot(rows, cols, i+1)
        ax2 = ax1.twinx() # create a twin y-axis on the right
        sns.pointplot(data=participant_df, x='block', y='correct', errorbar=('ci', 95), ax=ax1, color = 'k', scale = 0.5, errwidth=1.5)
        sns.pointplot(data=participant_df, x='block', y='Choice_Rule_Diff', ax=ax2, color='r', scale = 0.5, errwidth=1.5)

        if background_shading:
            # Smooth background shading for each block based on distribution
            prev_color = None
            start_block = None
            end_block = None
            for block in participant_df['block'].unique():
                block_df = participant_df[participant_df['block'] == block]
                if len(block_df) > 0:
                    dist = block_df['Distribution'].values[0]
                    color = color_dict[dist]

                    if prev_color is None:
                        prev_color = color
                        start_block = block
                        end_block = block
                    elif prev_color != color or block != end_block + 1:
                        ax1.axvspan(start_block - 0.5, end_block + 0.5, alpha=0.2, color=prev_color)
                        prev_color = color
                        start_block = block
                    end_block = block

            ax1.axvspan(start_block - 0.5, end_block + 0.5, alpha=0.2, color=prev_color)

        ax1.set_title(f"Participant ID: {participant_id}")
        ax1.set_ylim(0, 1)
        ax2.axhline(y=0, linestyle='--', color='k')

        ax2.set_ylim(-1, 1)
        ax2.set_yticks(np.arange(-1,1.1,0.4))
        ax2.tick_params(axis='y', labelcolor='r')

        # Set xticks to show only one label every 5 ticks if there are more than 5 units on the x-axis
        xticks = participant_df['block'].unique()
        if len(xticks) > 20:
            xticks = np.arange(participant_df['block'].min(), participant_df['block'].max() + 1, 20)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(int(x)) if int(x)==x else '' for x in xticks])

        ax1.set_ylabel('Proportion Correct', color='k')
        ax1.set_xlabel('Block', color='k')
        ax2.set_ylabel('Bias', color='r')
    plt.tight_layout()
    plt.show()

def plot_performance_and_bias_mega_sub(df, participant_ids, save_path=None):
    """
    Plots the performance and bias for the specified participant IDs, pooled together in a mega-subject.

    Args:
        df (pandas.DataFrame): The data frame to use for the plot.
        participant_ids (list): A list of participant IDs to include in the plot.

    Returns:
        None
    """
    # Filter data for specified participants and remove trials with no response
    valid_participant_ids = check_valid_participant_ids(df, participant_ids)
    df = df[df['Participant_ID'].isin(participant_ids)]
    df = df[df['No_response'] == False].reset_index(drop=True)

    # Set plot size and style
    # sns.set(style='ticks')
    fig, ax1 = plt.subplots(figsize=(8, 8))

    # Plot accuracy data
    sns.pointplot(data=df, x='block', y='correct', color='black', scale=0.5, errwidth=2, ax=ax1)

    # Plot bias data on a secondary y-axis
    ax2 = ax1.twinx()
    sns.pointplot(data=df, x='block', y='Choice_Rule_Diff', color='red', scale=0.5, errwidth=2, ax=ax2)

    # Set y-axis limits and ticks
    ax1.set_ylim(0, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_yticks(np.arange(-1, 1.1, 0.4))

    # Set x-axis ticks
    xticks = df['block'].unique() 
    if len(xticks) > 20:
        xticks = np.arange(df['block'].min(), df['block'].max() + 1, 10)
    ax1.set_xticks(xticks)

    # Set axis labels and title
    ax1.set_xlabel('Block')
    ax1.set_ylabel('Proportion Correct', color='k')
    ax2.set_ylabel('Bias', color='red')
    ax1.set_title('Accuracy and Bias')

    # Set tick colors
    ax1.tick_params(axis='y', colors='black')
    ax2.tick_params(axis='y', colors='red')

    # Add legend
    ax1.legend(['Accuracy'], loc='lower left')
    ax2.legend(['Bias'], loc='lower right')

    # Add 1 to all x-values (for 1 indexing)
    # plt.gca().set_xticklabels([str(int(label.get_text()) + 1) for label in ax1.get_xticklabels()])

    plt.tight_layout()
    # Save figure to a PDF file if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def check_participants(df, participant_ids):
    valid_ids = df.Participant_ID.unique()
    valid_participant_ids = []
    invalid_participant_ids = []
    for pid in participant_ids:
        if pid in valid_ids:
            valid_participant_ids.append(pid)
        else:
            invalid_participant_ids.append(pid)
    if invalid_participant_ids:
        print(f"Warning: Participant IDs {invalid_participant_ids} not found in dataframe")
    return valid_participant_ids

def psycho_fit(df, x='stim_relative_binned', y='choice'):
    choice_data = df.groupby(x)[y].agg(['mean', 'count', 'var', 'std', 'sem'])
    x_data = choice_data.index
    y_data = choice_data['mean'].values
    # Remove NaN values from x_data and y_data
    x_data_cleaned = [x for x, y in zip(x_data, y_data) if not np.isnan(y)]
    y_data_cleaned = [y for y in y_data if not np.isnan(y)]
    popt, pcov = curve_fit(psychometric, x_data_cleaned, y_data_cleaned, bounds=([-1., 0.01, 0., 0.], [1., 10, .5, .5]))
    return popt, pcov

def psycho_plot(df, popt, label, color, ax, x='stim_relative_binned', y='choice', scatter=True, legend=True):
    choice_data = df.groupby(x)[y].agg(['mean', 'count', 'var', 'std', 'sem'])
    x_data = choice_data.index
    if scatter == True:
        y_data = choice_data['mean'].values
        ax.scatter(x_data, y_data, color=color, s=5)
        ax.errorbar(x_data, choice_data['mean'], yerr=choice_data['sem']*1.96, 
                    fmt='.', markersize=2, color=color)
    x_lim = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = psychometric(x_lim, *popt)
    ax.plot(x_lim, y_fit, label=label, color=color, linewidth=1)
    ax.set_ylim(-0.05,1.05)
    ax.set_xlim(-1,1)  
    ax.set_xticks(np.linspace(-1, 1, 5))
    # ax.legend(prop={'size': 10})
    # ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Distance from Boundary')
    ax.set_ylabel('Ratio chose B')
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if legend == True:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))  # Adjust the bbox_to_anchor parameter

def psycho_plot_mega_sub(df, participant_ids, distributions, 
                         color_dict={'Uniform': '#2F9A76', 'Asym_left': '#007dc7', 'Asym_right': '#c23f63'}, 
                         plot_learning_gradient = False,
                         cmap_dict = {'Uniform': mcolors.LinearSegmentedColormap.from_list('greens', ['#B2DF8A', '#006400']),
                                      'Asym_left': mcolors.LinearSegmentedColormap.from_list('blues', ['#ADD8E6', '#000080']),
                                      'Asym_right': mcolors.LinearSegmentedColormap.from_list('reds', ['#FFB6C1', '#8B0000'])},
                         stim_dur_separate = False,
                         title='Mega-Subject', 
                         manual_legend = None,
                         legend = True,
                         save_path = None):
    """Plot psychometric functions for multiple participants and distributions."""
    participant_ids = check_participants(df, participant_ids)
    filtered_df = df[df['Distribution'].isin(distributions)]
    filtered_df = filtered_df[filtered_df['Participant_ID'].isin(participant_ids)]
    filtered_df = filtered_df[filtered_df.No_response == False]

    fig, ax = plt.subplots(figsize=(8,8))

    for distribution in distributions:
        dist_df = filtered_df[filtered_df['Distribution'] == distribution]
        if plot_learning_gradient == True:
            cmap = cmap_dict[distribution]
            colors = [cmap(i) for i in np.linspace(0, 1, len(dist_df.block.unique()))]
            for i, block in zip(range(0, len(dist_df.dist_block.unique())+1), dist_df.dist_block.unique()):
                if block % 10 == 0:
                    block_df = dist_df[dist_df.dist_block == block]
                    popt = psycho_fit(block_df)[0]
                    psycho_plot(block_df, popt, distribution + ' ' + str(block), 
                                colors[i], ax=ax, legend=False)
        else:
            if stim_dur_separate == True:
                for i, stim_dur in zip(range(0, len(dist_df.Stim_Dur.unique())+1), dist_df.Stim_Dur.unique()):
                    stum_dur_df = dist_df[dist_df.Stim_Dur == stim_dur]
                    popt = psycho_fit(stum_dur_df)[0]
                    psycho_plot(stum_dur_df, popt, distribution + ' ' + str(stim_dur)[0:3] + 'ms', 
                                color_dict[distribution][i], ax=ax, legend=False)
            else:
                popt = psycho_fit(dist_df)[0]
                psycho_plot(dist_df, popt, distribution, 
                            color_dict[distribution], ax=ax, legend=False)
    plt.title(title)

    # Add manual legend if manual_legend is not None
    if manual_legend:
        handles = []
        labels = []
        for key in manual_legend:
            handles.append(plt.Line2D([0], [0], color=manual_legend[key], linewidth=2))
            labels.append(key)
        ax.legend(handles, labels, bbox_to_anchor=(0, 1.0), loc='upper left', borderaxespad=0.2)

        # Add padding for legend
        fig.subplots_adjust(bottom=0.2)

    # Add legend if legend=True
    elif legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)

        # Add padding for legend
        fig.subplots_adjust(bottom=0.2)

    plt.tight_layout()

    # Save figure to a PDF file if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def psycho_plot_individual(df, participant_ids, distributions, 
                               color_dict={'Uniform': '#2F9A76', 'Asym_left': '#007dc7', 'Asym_right': '#c23f63'}, 
                               plot_learning_gradient = False,
                               cmap_dict = {'Uniform': mcolors.LinearSegmentedColormap.from_list('greens', ['#B2DF8A', '#006400']),
                                            'Asym_left': mcolors.LinearSegmentedColormap.from_list('blues', ['#ADD8E6', '#000080']),
                                            'Asym_right': mcolors.LinearSegmentedColormap.from_list('reds', ['#FFB6C1', '#8B0000'])},
                                stim_dur_separate = False,
                                title='Individual Subjects'):
    participant_ids = check_participants(df, participant_ids)
    filtered_df = df[df['Distribution'].isin(distributions)]
    filtered_df = filtered_df[filtered_df['Participant_ID'].isin(participant_ids)]
    filtered_df = filtered_df[filtered_df.No_response == False]
    
    max_cols = 3
    num_participants = len(participant_ids)
    cols = min(num_participants, max_cols)
    rows = int(np.ceil(num_participants / cols))
    
    fig = plt.figure(figsize=(cols*6, rows*4))

    if num_participants == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        axs = [ax]
    else:
        fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        axs = axs.flatten()[:num_participants]

    for i, participant_id in enumerate(participant_ids):
        if participant_id in df.Participant_ID.unique(): 
            ax = axs[i]
            participant_df = filtered_df[filtered_df.Participant_ID == participant_id]

            for distribution in participant_df.Distribution.unique():
                dist_df = participant_df[participant_df.Distribution == distribution]
                if plot_learning_gradient == True:
                    cmap = cmap_dict[distribution]
                    colors = [cmap(i) for i in np.linspace(0, 1, len(dist_df.block.unique()))]
                    for i, block in zip(range(0, len(dist_df.dist_block.unique())+1), dist_df.dist_block.unique()):
                        if block % 5 == 0:
                            block_df = dist_df[dist_df.dist_block == block]
                            popt = psycho_fit(block_df)[0]
                            psycho_plot(block_df, popt, distribution + ' ' + str(block), 
                                        colors[i], ax=ax)
                else:
                    if stim_dur_separate == True:
                        for i, stim_dur in zip(range(0, len(dist_df.Stim_Dur.unique())+1), dist_df.Stim_Dur.unique()):
                            stum_dur_df = dist_df[dist_df.Stim_Dur == stim_dur]
                            popt = psycho_fit(stum_dur_df)[0]
                            psycho_plot(stum_dur_df, popt, distribution + ' ' + str(stim_dur), 
                                        color_dict[distribution][i], ax=ax)
                    else:
                        popt = psycho_fit(dist_df)[0]
                        psycho_plot(dist_df, popt, distribution, color_dict[distribution], ax=ax)
            ax.set_title(f"Participant ID: {participant_id}")
    fig.suptitle(title, size=15)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()
    plt.show()

def gen_blockwise_psych_params(df, participant_ids):
    '''
    Generate blockwise psychometric parameters and errors for these parameters for each participant
    Args:
        df: dataframe with all the data
        participant_ids: list of participant ids
    Returns:
        psych_params_df: dataframe with blockwise psychometric parameters and errors for each participant
    '''
    df = df[df.No_response == False]
    psych_params_list = []

    for participant in df.Participant_ID.unique():
        participant_df = df[df.Participant_ID == participant]
        psych_params = []
        
        for block in participant_df.block.unique():
            block_df = participant_df[participant_df.block == block]
            try:
                popt, pcov = psycho_fit(block_df)   
            except:
                popt = [np.nan, np.nan, np.nan, np.nan]
                pcov = [[np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan]]
                print(f"Warning: Could not fit psychometric function for participant {participant} block {block}")
            psych_params.append([participant, block, block_df.Distribution.values[0], *popt, *np.sqrt(np.diag(pcov))])
        
        psych_params_list.extend(psych_params)

    psych_params_df = pd.DataFrame(psych_params_list, columns=['Participant_ID', 'Block', 'Distribution', 'Block_Means', 
                                                               'Block_Slopes', 'Block_Lapses_1', 'Block_Lapses_2', 
                                                               'Block_Means_Std', 'Block_Slopes_Std', 'Block_Lapses_1_Std', 
                                                               'Block_Lapses_2_Std'])
    return psych_params_df

def plot_blockwise_psych_params(df, participant_ids, 
                                distributions = ['Uniform', 'Asym_left', 'Asym_right'],
                                color_dict = {'Uniform': '#2F9A76', 'Asym_left': '#007dc7', 'Asym_right': '#c23f63'}):
    '''
    Plot blockwise psychometric parameters for chosen participants and distributions
    Args:
        df: dataframe with all the data
        participant_ids: list of participant ids
        distributions: list of distributions
        color_dict: dictionary with colors for each distribution
    Returns:
        fig: figure with blockwise psychometric parameters for each participant and distribution
    '''
    # Check that all specified participant IDs are valid
    valid_participant_ids = check_valid_participant_ids(df, participant_ids)

    # Create subplots
    fig, axs = plt.subplots(len(valid_participant_ids), 4, figsize=(16, 4*len(valid_participant_ids)), sharex=False)

    # Loop through each participant and plot data
    for i, participant in enumerate(valid_participant_ids):
        # Filter data for the current participant
        participant_df = df[df.Participant_ID == participant]

        # Create empty dictionary for labels
        labels = {}

        # Loop through each distribution and plot data with error bars
        for j, dist in enumerate(distributions):
            dist_df = participant_df[participant_df.Distribution == dist]
            color = color_dict[dist]

            # Connect only adjacent dots in the plots
            for k in range(1, len(dist_df.Block)):
                if dist_df.Block.iloc[k] - dist_df.Block.iloc[k-1] == 1:
                    # Plot adjacent block_Means with error bars
                    axs[i, 0].plot(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Means.iloc[k-1:k+1], 'o-', label=dist, color=color, markersize=3)
                    axs[i, 0].errorbar(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Means.iloc[k-1:k+1], yerr=dist_df.Block_Means_Std.iloc[k-1:k+1], color=color, alpha=0.5)
                    axs[i, 0].set_title('{} - Mean'.format(participant))
                    axs[i, 0].set_ylim(-1, 1)  # Set y-limits

                    # Plot adjacent block_Slopes with error bars
                    axs[i, 1].plot(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Slopes.iloc[k-1:k+1], 'o-', label=dist, color=color, markersize=3)
                    axs[i, 1].errorbar(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Slopes.iloc[k-1:k+1], yerr=dist_df.Block_Slopes_Std.iloc[k-1:k+1], color=color, alpha=0.5)
                    axs[i, 1].set_title('{} - Sigma'.format(participant))
                    axs[i, 1].set_ylim(-1, 1)  # Set y-limits


                    # Plot adjacent block_Lapses_1 with error bars
                    axs[i, 2].plot(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Lapses_1.iloc[k-1:k+1], 'o-', label=dist, color=color, markersize=3)
                    axs[i, 2].errorbar(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Lapses_1.iloc[k-1:k+1], yerr=dist_df.Block_Lapses_1_Std.iloc[k-1:k+1], color=color, alpha=0.5)
                    axs[i, 2].set_title('{} - Lapse_1'.format(participant))
                    axs[i, 2].set_ylim(-1, 1)  # Set y-limits

                    # Plot adjacent block_Lapses_2 with error bars
                    axs[i, 3].plot(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Lapses_2.iloc[k-1:k+1], 'o-', label=dist, color=color, markersize=3)
                    axs[i, 3].errorbar(dist_df.Block.iloc[k-1:k+1], dist_df.Block_Lapses_2.iloc[k-1:k+1], yerr=dist_df.Block_Lapses_2_Std.iloc[k-1:k+1], color=color, alpha=0.5)
                    axs[i, 3].set_title('{} - Lapse_2'.format(participant))
                    axs[i, 3].set_ylim(-1, 1)  # Set y-limits

                    # Add label to dictionary for legend
                    if dist not in labels:
                        labels[dist] = color

    # Add legend to first subplot
    handles = [mpatches.Patch(color=color_dict[label], label=label) for label in color_dict]
    axs[0, 0].legend(handles=handles, fontsize=10)

    # Add x-label to bottom plots
    for ax in axs[-1, :]:
        ax.set_xlabel('Block')

    # Adjust spacing and show plot
    plt.tight_layout()
    plt.show()
    
def format_data_hmmglm(df, Participant_ID, input_dim, 
                       feature = 'stim_relative_binned',
                       bias = 1,
                       target = 'choice'):
    '''
    Format data for HMM-GLM
    Args:
        df: dataframe with all the data
        Participant_ID: participant ID
        input_dim: number of input dimensions
        feature: feature to use for input
        bias: bias to add to input
        target: target to predict
    Returns:
        inpts: list of inputs
        true_choices: list of true choices
        animal_df: dataframe with data for the current participant
    '''
    animal_df = df[df['Participant_ID'] == Participant_ID]
    animal_df = animal_df[animal_df['No_response'] == False]
    stim_vals = []
    for block in animal_df['block'].unique():
        stim_vals.append(animal_df[animal_df['block'] == block][feature].values.to_numpy())

    true_choices = []
    for block in animal_df['block'].unique():
        block_choices = animal_df[animal_df['block'] == block][target].astype(int).values
        true_choices.append(block_choices)
    true_choices = [np.transpose([array]) for array in true_choices]

    num_sess = len(true_choices) # number of sessions
    num_trials_per_sess = [len(choice) for choice in true_choices] # number of trials per session

    inpts = [] # initialize inpts list
    for sess in range(num_sess):
        num_trials = num_trials_per_sess[sess]
        sess_inpts = np.ones((num_trials, input_dim))
        sess_inpts[:, 0] = stim_vals[sess]
        sess_inpts[:, 1] = bias
        inpts.append(sess_inpts)
    return inpts, true_choices, animal_df

def add_previous_trial_data(group):
    """
    Add previous trial data (stim_relative_binned, choice, and correct) to the DataFrame for each participant.
    
    Parameters
    ----------
    group : pd.DataFrame
        The data for a single participant, assumed to have the columns 'Trial', 'No_response',
        'stim_relative_binned', 'choice', and 'correct'.
        
    Returns
    -------
    group : pd.DataFrame
        The modified input DataFrame with additional columns for previous trial data
        ('stim_relative_1', 'choice_1', and 'correct_1').
    """
    # Shift the stim_relative_binned, choice, and correct columns by 1 row to get previous trial data
    group['stim_relative_1'] = group['stim_relative_binned'].shift(1)
    group['choice_1'] = group['choice'].shift(1)
    group['correct_1'] = group['correct'].shift(1)

    # Set the new columns to NaN for the first trial of each participant
    group.loc[group['Trial'] == 0, ['stim_relative_1', 'choice_1', 'correct_1']] = np.nan
    
    # Set choice_1 and correct_1 to NaN for trials where the previous trial had no response
    group.loc[group['No_response'].shift(1) == True, ['choice_1', 'correct_1']] = np.nan

    return group

def compute_update_matrix(df, prev_correct=True):
    """
    Computes the update matrix based on data_df and pre_correct parameters.

    Parameters:
    data_df (pandas.DataFrame): A pandas DataFrame containing the data.
    pre_correct (bool): A boolean value indicating whether the previous response was correct.

    Returns:
    numpy.ndarray: An update matrix with shape (num_bins, num_bins)
    """
    num_bins = len(df.stim_relative_binned.unique())
    update_matrix = np.zeros((num_bins, num_bins))
    prev_correct = 1 if prev_correct else 0
    prev_correct_df = df[df.correct_1 == prev_correct]

    for column, prev_stim in enumerate(sorted(prev_correct_df.stim_relative_1.unique())):
        prev_stim_df = prev_correct_df[prev_correct_df.stim_relative_1 == prev_stim]
        for row, curr_stim in enumerate(sorted(prev_correct_df.stim_relative_binned.unique())):
            curr_stim_df = prev_stim_df[prev_stim_df.stim_relative_binned == curr_stim]
            curr_stim_avg_df = prev_correct_df[prev_correct_df.stim_relative_binned == curr_stim]
            val = curr_stim_df.choice.mean() - curr_stim_avg_df.choice.mean()
            # val is difference between conditional psych at current stim and grand average at current stim
            update_matrix[row, column] = val

    update_matrix = np.flip(update_matrix, 0)
    return update_matrix

def plot_update_matrix(update_matrix, title='', save_path=None):
    """
    Plots a heatmap based on the update matrix.

    Parameters:
    update_matrix (numpy.ndarray): An update matrix with shape (num_bins, num_bins).
    title (str): A string to use as the title.

    Returns:
    None
    """
    num_bins = update_matrix.shape[0]
    cvals = [-1, 0, 1]
    colors = ['darkorange', 'white', 'blueviolet']
    norm2 = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm2, cvals), colors))
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('', tuples)

    fig, ax = plt.subplots(figsize=(10, 8))

    g = sns.heatmap(update_matrix, cmap=cmap2, vmin=-0.15, vmax=0.15)
    g.set_xticks(np.arange(0.5, num_bins + 1, 3.5), np.arange(-1, 1.5, 1))
    g.set_yticks(np.arange(0.5, num_bins + 1, 3.5), np.arange(1, -1.5, -1))

    plt.title(title)
    plt.xlabel('Previous Stimulus')
    plt.ylabel('Current Stimulus')

    cbar = g.collections[0].colorbar
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Updating %', rotation=270)

    plt.tight_layout()
    # Save figure to a PDF file if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_choice_updating(update_matrix, title=None):
    # Extract the rows of interest
    easy_rows_indices = [0, 1, 6, 7]
    easy_rows = update_matrix[easy_rows_indices, :]

    hard_rows_indices = np.arange(2,6,1)
    hard_rows = update_matrix[hard_rows_indices, :]

    easy_average = np.nanmean(easy_rows[:, 0:len(update_matrix)], axis=0)
    ax = sns.pointplot(x=np.arange(0, len(update_matrix),1), 
                       y=easy_average,
                       color='gray')

    hard_average = np.nanmean(hard_rows[:, 0:len(update_matrix)], axis=0)
    sns.pointplot(x=np.arange(0, len(update_matrix),1), 
                  y=hard_average,
                  color='k',
                  ax=ax)

    ax.set_ylim(-0.2,0.2)
    ax.set_yticks(np.arange(-0.2,0.3,0.2))
    ax.set_xticks(np.arange(0,len(update_matrix)+1,2))
    ax.set_xticklabels(np.arange(-1,1.1,0.5)) 
    ax.set_xlabel('Previous Stimulus')
    ax.set_ylabel('Updating %')
    ax.axhline(y=-0.1, linestyle='--', color='gray', alpha=0.5)
    ax.axhline(y=0.1, linestyle='--', color='gray', alpha=0.5)

    ax.legend(handles=[mpatches.Patch(color='gray', label='Easy', linewidth=0.5), 
                       mpatches.Patch(color='black', label='Hard', linewidth=1)],
              title='Current Choice', title_fontsize='large')

    if title is not None:
        ax.set_title(title, fontsize=22)

    plt.show()

def remove_blocks(df, participant_id, blocks_to_remove):
    # Remove specified blocks for the participant
    df = df[~((df['Participant_ID'] == participant_id) & (df['block'].isin(blocks_to_remove)))]

    # Reindex the 'block' column for the participant
    mask = df['Participant_ID'] == participant_id
    unique_blocks = df.loc[mask, 'block'].unique()
    reindex_mapping = {block: i for i, block in enumerate(unique_blocks)}

    df.loc[mask, 'block'] = df.loc[mask, 'block'].map(reindex_mapping)
    
    return df

def add_dist_sequence_number(clean_psych_params_df):
    """
    Add sequence column that tracks the number of distribution switches.

    Parameters:
        clean_psych_params_df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
        pandas.DataFrame: The DataFrame with an additional 'Sequence' column.
    """
    
    # Initialize the sequence number for each participant and distribution to NaN
    clean_psych_params_df.loc[:, 'Sequence'] = pd.Series(dtype='Int64')

    # Group the dataframe by Participant_ID and Distribution, and assign the sequence number for each group
    for (participant, distribution), group in clean_psych_params_df.groupby(['Participant_ID', 'Distribution']):
        indices = group.index
        seq_nums = [0]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                seq_nums.append(seq_nums[-1])
            else:
                seq_nums.append(seq_nums[-1] + 1)
        clean_psych_params_df.loc[indices, 'Sequence'] = seq_nums

    return clean_psych_params_df

def plot_mean_shift_Vs_sigma_individual(df, participants, pre_dist, post_dist, pre_sequence, post_sequence, average_pre_switch = True):
    """
    Plots the mean shift against sigma for each participant.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be processed.
        participants (list): A list of participant IDs.
        pre_dist (str): The distribution before the switch.
        post_dist (str): The distribution after the switch.
        pre_sequence (int): The sequence number before the switch.
        post_sequence (int): The sequence number after the switch.

    Returns:
        None
    """

    # Calculate the number of rows required
    n_rows = int(np.ceil(len(participants) / 3))

    # Create a new figure for all participants
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5))
    axs = axs.flatten()

    # Loop through each participant
    for ax, participant in zip(axs, participants):
        # Subset the dataframe to only include the current participant
        df_participant = df[df['Participant_ID'] == participant]

        # Define a colormap
        cmap = plt.cm.viridis

        # Get the unique blocks for the current participant, post distribution, and post sequence
        unique_blocks = df_participant[(df_participant['Distribution'] == post_dist) & (df_participant['Sequence'] == post_sequence)]['Block'].unique()

        # Loop through each block for the current participant
        for idx, block_num in enumerate(unique_blocks):
            if average_pre_switch == True:
                # Compute the difference between average pre_dist, pre sequence, and post_dist post sequence for the current block 
                diff = (
                    np.mean(
                        df_participant[
                            (df_participant['Distribution'] == pre_dist)
                            & (df_participant['Sequence'] == pre_sequence)
                        ]['Block_Means']
                    )
                    - df_participant[
                        (df_participant['Distribution'] == post_dist)
                        & (df_participant['Sequence'] == post_sequence)
                        & (df_participant['Block'] == block_num)
                    ]['Block_Means']
                )
            else:
                pre_switch_block = df_participant[(df_participant['Distribution'] == pre_dist) & (df_participant['Sequence'] == pre_sequence)]['Block'].max()
                last_mean = df_participant[(df_participant['Distribution'] == pre_dist) & (df_participant['Sequence'] == pre_sequence) & (df_participant['Block'] == pre_switch_block)]['Block_Means'].values[0]
                diff = last_mean - df_participant[(df_participant['Distribution'] == post_dist) & (df_participant['Sequence'] == post_sequence) & (df_participant['Block'] == block_num)]['Block_Means']

            # Get the slope for the current block post sequence
            slope = df_participant[(df_participant['Block'] == block_num) & (df_participant['Sequence'] == post_sequence)]['Block_Slopes']

            # Define a norm to map the block numbers to different colors for the current participant
            norm = plt.Normalize(vmin=0, vmax=len(unique_blocks))

            # Plot diff against slope and add text with the Block number and color code
            ax.scatter(slope, diff, color=cmap(norm(idx)))
            ax.set_xlabel(r'Sigma$_i$')
            ax.set_ylabel(fr'$\overline{{{pre_dist}}}$ - {post_dist}$_i$')
            ax.set_title(f'Participant {participant}')
            ax.text(slope, diff+0.02, '', ha='center', va='bottom', color=cmap(norm(idx)))

            # Set x and y limits
            ax.set_xlim([-0.05, 1])
            ax.set_ylim([-0.8, 0.8])

            # Set x and y ticks
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-0.5, 0, 0.5])
                           
            # Add horizontal line at y = 0
            ax.axhline(y=0, color='k', linestyle='--')

        # Add a colorbar to show the block number to color mapping
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set_title('Block post \n switch (i)', y=1.02)

        # Set the format of the colorbar legend to integer values
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add super title
    fig.suptitle(pre_dist + ' ' +  str(pre_sequence) + ' --> ' +  post_dist + ' ' + str(post_sequence), size=16)

    # Remove empty subplots if there are any
    for i in range(len(participants), n_rows * 3):
        fig.delaxes(axs[i])

    # Show the plot for all participants
    plt.tight_layout()
    plt.show()

def plot_psycho_params(df, participant_ids, 
                       distributions = ['Uniform', 'Asym_left', 'Asym_right'], 
                       color_dict = {'Uniform': '#2F9A76', 'Asym_left': '#007dc7', 'Asym_right': '#c23f63'}, 
                       manual_legend = None,
                       legend = True,
                       title = 'Psychometric Parameters',
                       stim_dur_separate = False,
                       save_path = None):
    participant_ids = check_participants(df, participant_ids)
    df = df[df.Participant_ID.isin(participant_ids)]
    df = df[df.Distribution.isin(distributions)]

    fig, ax = plt.subplots(figsize=(8,8))

    n = -0.1
    for distribution in df.Distribution.unique():
        dist_df = df[df.Distribution == distribution]
        if stim_dur_separate == True:
            for i, stim_dur in zip(range(0, len(dist_df.Stim_Dur.unique())+1), dist_df.Stim_Dur.unique()):
                stim_dur_df = dist_df[dist_df.Stim_Dur == stim_dur]
                popt, pcov = psycho_fit(stim_dur_df)
                x = np.arange(1,5)
                y = popt
                y_err = np.sqrt(np.diag(pcov))
                plt.errorbar(x+n, y, yerr=y_err, fmt='o', label=distribution, color=color_dict[distribution][i], capsize=3)
                n += 0.1
        else:
            popt, pcov = psycho_fit(dist_df)
            x = np.arange(1,5)
            y = popt
            y_err = np.sqrt(np.diag(pcov))
            plt.errorbar(x+n, y, yerr=y_err, fmt='o', label=distribution, color=color_dict[distribution], capsize=3)
            n += 0.1

    
    # Add manual legend if manual_legend is not None
    if manual_legend:
        handles = []
        labels = []
        for key in manual_legend:
            marker = 'o'
            color = manual_legend[key]
            line = plt.Line2D([0], [0], color=color, marker=marker, linestyle='None')
            handles.append(line)
            labels.append(key)
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.2)
        # Add padding for legend
        fig.subplots_adjust(bottom=0.2)

    elif legend:
        plt.legend()
    xticks = ['Mean', 'Sigma', 'Lapse 1', 'Lapse 2']
    plt.xticks(np.arange(1,5), xticks)
    plt.yticks(np.arange(0,0.6,0.1))
    plt.axhline(y=0, linestyle='--', color='k')
    plt.xlabel(' ')
    plt.ylabel('Parameter Value')
    plt.title(title)
    plt.tight_layout()
    # Save figure to a PDF file if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_sim_unif_asymR_shfit(sigma, mu = 0, lapse1 = 0, lapse2 = 0, 
                              color_dict = {'Uniform': '#2F9A76', 'Asym_left': '#007dc7', 'Asym_right': '#c23f63'},
                              shade_20_80=True, legend=True, title=None,
                              save_path = None):
    x = np.linspace(-1, 1, 100)
    y = psychometric(x, mu=mu, sigma=sigma, lapse1=lapse1, lapse2=lapse2)

    x_20, x_80 = x[np.argmin(np.abs(y - 0.2))], x[np.argmin(np.abs(y - 0.8))]

    y_asymR = psychometric(x - x_80, mu=mu, sigma=sigma, lapse1=lapse1, lapse2=lapse2)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y, label='Uniform', color=color_dict['Uniform'])
    # plot another psychometric with mean shifted to the right by x_80
    ax.plot(x, y_asymR, label='Asym_right', color=color_dict['Asym_right'])

    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Distance from Boundary')
    ax.set_ylabel('Ratio chose B')
    # add x and y ticks
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # add vertical line at 0
    ax.axvline(0, color='k', linewidth=1, linestyle='--')
    ax.set_title(title)

    if shade_20_80:
        ax.axvspan(x_20, x_80, alpha=0.2, color='k')

    if legend:
        ax.legend(handles=[Line2D([0], [0], color='gray', linewidth=10, label='<80% \nRegion'), 
                           Line2D([0], [0], color= color_dict['Uniform'], linewidth=1, label='Uniform'),
                           Line2D([0], [0], color = color_dict['Asym_right'], linewidth=1, label='Hard-A')],
                           bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.2)

    # add an arrow on the Uniform curve pointing towards the right
    arrow_x = 0
    arrow_y = 0.5
    arrow_dx = arrow_x + x_80
    arrow_dy = 0
    ax.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, color='#c23f63', width=0.005, head_width=0.03, length_includes_head=True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_mean_shift_Vs_sigma(df, participants, markers, pre_dist, post_dist, pre_sequence, post_sequence, color, ax, title = None):
    """
    Plots the mean shift versus sigma for a group of participants.
    
    Args:
    - df: a pandas DataFrame with the experimental data
    - participants: a list of participant IDs to include in the plot
    - markers: a list of markers to use for each participant
    - pre_dist: the distribution used in the pre-transition
    - post_dist: the distribution used in the post-transition
    - pre_sequence: the sequence number used in the pre-transition
    - post_sequence: the sequence number used in the post-transition
    - color: the color to use for the scatter plot
    - ax: the matplotlib axes to plot on
    """

    # Create a dictionary to map participants to markers
    marker_dict = dict(zip(participants, markers))
    color_dict = {pre_dist + ' - ' + post_dist: color}

    # Loop through each participant in the group
    for participant in participants:
        # Subset the dataframe to only include the current participant
        df_participant = df[df['Participant_ID'] == participant]

        # Compute the mean shift shift and slope for the participant
        mean_shift = np.mean(df_participant[
                            (df_participant['Distribution'] == pre_dist)
                          & (df_participant['Sequence'] == pre_sequence)
        ]['Block_Means']) - np.mean(df_participant[
                                   (df_participant['Distribution'] == post_dist)
                                 & (df_participant['Sequence'] == post_sequence)
        ]['Block_Means'])
        slope = df_participant[df_participant['Sequence'] == post_sequence]['Block_Slopes'].mean()

        # Plot the global difference versus slope for the participant
        ax.scatter(slope, mean_shift, marker=marker_dict[participant], color=color, label=participant, s=100)

        # Loop through each block for the participant
        unique_blocks = df_participant[
                       (df_participant['Distribution'] == post_dist)
                     & (df_participant['Sequence'] == post_sequence)
        ]['Block'].unique()
        for idx, block_num in enumerate(unique_blocks):
            # Compute the difference for the current block
            block_mean_shift = np.mean(df_participant[
                                      (df_participant['Distribution'] == pre_dist)
                                    & (df_participant['Sequence'] == pre_sequence)
            ]['Block_Means']) - df_participant[
                               (df_participant['Distribution'] == post_dist)
                             & (df_participant['Sequence'] == post_sequence)
                             & (df_participant['Block'] == block_num)
            ]['Block_Means']

            # Get the slope for the current block and post sequence
            slope = df_participant[
                   (df_participant['Block'] == block_num)
                 & (df_participant['Sequence'] == post_sequence)
            ]['Block_Slopes']

            # Plot diff against slope and add text with the block number and marker code
            ax.scatter(slope, block_mean_shift, marker=marker_dict[participant], color=color, alpha=0.2)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--')

    # set axis limits
    ax.set_xlim([-0.05, 1])
    ax.set_ylim([-1, 1])

    # Set x and y ticks
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Set the axis labels and legend
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Mean Shift')
    ax.set_title(title)    

def plot_mean_shift_Vs_sigma_comp(cohort_dict_1, cohort_dict_2, legend = True, manual_legend = None, save_path = None):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    plot_mean_shift_Vs_sigma(cohort_dict_1['df'],
                                participants=cohort_dict_1['participants'],
                                markers=cohort_dict_1['markers'],
                                pre_dist=cohort_dict_1['pre_dist'],
                                post_dist=cohort_dict_1['post_dist'],
                                pre_sequence=cohort_dict_1['pre_sequence'],
                                post_sequence=cohort_dict_1['post_sequence'],
                                color=cohort_dict_1['color'],
                                ax=ax,
                                title=cohort_dict_1['title'])
    
    plot_mean_shift_Vs_sigma(cohort_dict_2['df'],
                                participants=cohort_dict_2['participants'],
                                markers=cohort_dict_2['markers'],
                                pre_dist=cohort_dict_2['pre_dist'],
                                post_dist=cohort_dict_2['post_dist'],
                                pre_sequence=cohort_dict_2['pre_sequence'],
                                post_sequence=cohort_dict_2['post_sequence'],
                                color=cohort_dict_2['color'],
                                ax=ax,
                                title=cohort_dict_2['title'])
    
    if legend:
        color_dict = {
            cohort_dict_1['pre_dist'] + ' - ' + cohort_dict_1['post_dist']: cohort_dict_1['color'],
            cohort_dict_2['pre_dist'] + ' - ' + cohort_dict_2['post_dist']: cohort_dict_2['color']
        }        
        marker_dict = dict(zip(cohort_dict_1['participants'], cohort_dict_1['markers']))
        marker_dict.update(dict(zip(cohort_dict_2['participants'], cohort_dict_2['markers'])))
        
        if manual_legend is not None:
            color_dict = {manual_legend.get(k, k): v for k, v in color_dict.items()}
        
        color_handles = [plt.Line2D([], [], marker='o', linestyle='None', color=color_dict[color], label=color) for color in color_dict.keys()]
        color_legend = ax.legend(handles=color_handles, loc='lower left', title='Transition')
        ax.add_artist(color_legend)
        
        marker_handles = [plt.Line2D([], [], marker=marker_dict[marker], linestyle='None', color='black', label=str(marker)) for marker in marker_dict.keys()]
        marker_legend = ax.legend(handles=marker_handles, loc='upper right', title='Participant_ID')
        ax.add_artist(marker_legend)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_glmhmm_weights(glmhmm, num_states, input_dim,
                        color_dict, 
                        save_path = None):
    """
    Plot the weights of the GLM-HMM
    Args:
        glmhmm (GLMHMM): GLM-HMM model
        num_states (int): number of states
        input_dim (int): number of covariates
        color_dict (dict): dictionary mapping state number to color
        save_path (str): path to save the figure
    Returns:
        None
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))

    recovered_weights = glmhmm.observations.params
    for k in range(num_states):
        ax.plot(range(input_dim), recovered_weights[k][0], color=color_dict[k],
                 lw=1.5,  label='State ' + str(k+1), linestyle='-')
    ax.tick_params(axis='y')
    ax.set_ylabel("GLM weight")
    ax.set_xlabel("Covariate")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Stimulus', 'Bias'])
    ax.axhline(y=0, color="k", alpha=0.5, ls="--")
    ax.legend()
    ax.invert_yaxis()  # Invert y-axis
    ax.set_yticks(np.arange(-8, 7, 2), np.arange(8, -7, -2))
    ax.set_title(' ')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_glmhmm_trans_mat(glmhmm, num_states, decimals=3, save_path=None):
    """
    Plot the transition matrix of a GLM-HMM
    Args:
        glmhmm (GLMHMM): GLM-HMM model
        num_states (int): number of states
        decimals (int): number of decimals to round to
        save_path (str): path to save the figure
    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    recovered_trans_mat = np.exp(glmhmm.transitions.log_Ps)
    im = ax.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(recovered_trans_mat.shape[0]):
        for j in range(recovered_trans_mat.shape[1]):
            text = ax.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=decimals)), ha="center", va="center",
                            color="k")
    ax.set_xlim(-0.5, num_states - 0.5)
    ax.set_xticks(range(0, num_states), range(1, num_states + 1))
    ax.set_xlabel("State t")

    ax.set_yticks(range(0, num_states), range(1, num_states + 1))
    ax.set_ylim(num_states - 0.5, -0.5)
    ax.set_title(" ")
    ax.set_ylabel("State t-1")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_by_state(animal_df, num_states, color_dict, save_path=None):
    """
    Plot the accuracy of a single animal by state
    Args:
        animal_df (pd.DataFrame): dataframe with animal data
        num_states (int): number of states
        color_dict (dict): dictionary mapping state number to color
        save_path (str): path to save the figure
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for state in range(num_states):
        state_df = animal_df[animal_df.state == state]
        correct = np.sum(state_df.correct)
        total = state_df.shape[0]
        accuracy = correct / total * 100
        # plot % correct in bar chart
        plt.bar(state + 1, accuracy, color=color_dict[state])
        plt.text(state + 1, accuracy, f"{accuracy:.1f}", ha="center", va="bottom")

    # also plot % correct for all trials
    total_correct = np.sum(animal_df.correct)
    total_trials = animal_df.shape[0]
    total_accuracy = total_correct / total_trials * 100
    plt.bar(0, total_accuracy, color='gray')
    plt.text(0, total_accuracy, f"{total_accuracy:.1f}", ha="center", va="bottom")

    plt.xticks(range(num_states + 1), ['All'] + [str(i+1) for i in range(num_states)])
    plt.yticks([0, 25, 50, 75, 100])
    plt.xlabel('State')
    plt.ylabel('Accuracy (%)')
    plt.ylim([50, 100])
    plt.title(' ')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_psycho_by_state(animal_df, num_states, color_dict, save_path=None):
    """
    Plot psychometric curves for each state
    Args:
        animal_df (pd.DataFrame): dataframe with animal data
        num_states (int): number of states
        color_dict (dict): dictionary mapping state number to color
        save_path (str): path to save the figure
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8))  # Set the figure size to 8x8 inches

    for state in range(num_states):
        state_df = animal_df[animal_df.state == state]
        psycho_plot(state_df, psycho_fit(state_df)[0], 
                       label='state ' + str(state + 1), 
                       color=color_dict[state], 
                       ax=ax, legend=False)

    plt.title(' ')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_frac_state_occupancies(state_occupancies, num_states, color_dict, save_path=None):
    '''
    Plots the fractional occupancies of each state in a bar plot   
    Args:
        state_occupancies (array): array of fractional occupancies of each state
        num_states (int): number of states
        color_dict (dict): dictionary of colors for each state
        save_path (str): path to save figure to
    Returns:
        None
        '''
    fig = plt.figure(figsize=(8, 8))
    for z, occ in enumerate(state_occupancies):
        plt.bar(z, occ, color=color_dict[z])
    
    plt.ylim((0, 1))
    plt.xticks(range(num_states), [str(i+1) for i in range(num_states)])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1']) 
    plt.xlabel('State')
    plt.ylabel('Fraction Occupancy')
    plt.title(' ')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_state_probabilities(posterior_probs, sess_id, num_states, color_dict, title=None, legend_bbox=None, save_path=None):
    """
    Plot the posterior probabilities of each state for a single session
    Args:
        posterior_probs (dict): dictionary of posterior probabilities for each session
        sess_id (int): session ID
        num_states (int): number of states
        color_dict (dict): dictionary mapping state number to color
        title (str): title of the plot
        legend_bbox (tuple): tuple of coordinates for legend
        save_path (str): path to save the figure
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8), sharex=False)

    for k in range(num_states):
        ax.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2, color=color_dict[k])
    ax.set_ylim((-0.01, 1.01))
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Trial #")
    ax.set_ylabel("p(state)")
    if legend_bbox:
        ax.legend(loc='center', bbox_to_anchor=legend_bbox)
    
    sess_posterior_probs = posterior_probs[sess_id]
    sess_max_posterior_probs = np.argmax(sess_posterior_probs, axis=1)
    state_change_indices = np.where(sess_max_posterior_probs[:-1] != sess_max_posterior_probs[1:])[0] + 1
    
    for i in range(len(state_change_indices)):
        ax.axvline(x=state_change_indices[i], color='k', alpha=0.5, ls='--')

    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_choice_scatter(animal_df, posterior_probs, sess_id, title=None, save_path=None, legend_bbox=None):
    """
    Plot the choices made by the animal for a single session
    Args:
        animal_df (pd.DataFrame): dataframe with animal data
        posterior_probs (dict): dictionary of posterior probabilities for each session
        sess_id (int): session ID
        title (str): title of the plot
        save_path (str): path to save the figure
        legend_bbox (tuple): tuple of coordinates for legend
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {1: ('gray', 'o'), 0: ('red', 'v')}
    x = animal_df[(animal_df['block'] == sess_id) & (~animal_df['No_response'])].reset_index(drop=True).index
    y = animal_df[(animal_df['block'] == sess_id) & (~animal_df['No_response'])]['choice']
    y_jitter = y + np.random.uniform(low=-0.05, high=0.05, size=len(y))
    correctness = animal_df[animal_df['block'] == sess_id]['correct']
    marker_styles = [colors[c][1] for c in correctness]
    for xi, yi, ci, mi in zip(x, y_jitter, [colors[c][0] for c in correctness], marker_styles):
        ax.scatter(xi, yi, c=ci, marker=mi, s=50, alpha=0.5)
    ax.set_yticks([0, 1])
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Choice')
    ax.set_yticklabels(['L', 'R'])

    sess_posterior_probs = posterior_probs[sess_id]
    sess_max_posterior_probs = np.argmax(sess_posterior_probs, axis=1)
    state_change_indices = np.where(sess_max_posterior_probs[:-1] != sess_max_posterior_probs[1:])[0] + 1

    for i in range(len(state_change_indices)):
        ax.axvline(x=state_change_indices[i], color='k', alpha=0.5, ls='--')

    if title:
        ax.set_title(title)

    if legend_bbox is not None:
        legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label='Correct', markerfacecolor='gray', markersize=10),
                           mlines.Line2D([0], [0], marker='v', color='w', label='Incorrect', markerfacecolor='red', markersize=10)]
        legend_args = {'handles': legend_elements, 'loc': 'center right'}
        legend_args['bbox_to_anchor'] = legend_bbox
        ax.legend(**legend_args)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_state_occ_by_dist(animal_df, distribution_order, color_dict, distribution_dict, save_path=None):
    """
    Plots the state occupancy for each distribution side by side.
    Args:
        animal_df (pd.DataFrame): The animal DataFrame.
        distribution_order (list): The desired order of distributions.
        color_dict (dict): The color dictionary.
        distribution_dict (dict): The distribution dictionary.
        save_path (str): The path to save the figure to.
    Returns:
        None
    """

    # Group the DataFrame by 'Distribution' and 'State' columns and get the counts
    freq = animal_df.groupby(['Distribution', 'state']).size().reset_index(name='Count')
    # Normalize the frequency data
    freq_norm = freq.groupby('Distribution')['Count'].transform(lambda x: x / x.sum())
    freq['Normalized_Count'] = freq_norm
    # Pivot the DataFrame to create a side-by-side bar chart-friendly format
    pivot_df = freq.pivot(index='Distribution', columns='state', values='Normalized_Count').fillna(0)
    # Reorder the pivot_df based on distribution_order
    pivot_df = pivot_df.reindex(distribution_order)
    # Create the 9 bars figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set the bar width
    bar_width = 0.2
    # Calculate the offset for each bar position
    bar_offset = np.arange(len(pivot_df))
    
    # Plot the bars for each state, side by side
    for i, state in enumerate(pivot_df.columns):
        ax.bar(bar_offset + (i * bar_width), pivot_df[state], width=bar_width, color=color_dict.get(i, '#333333'))
    
    # Add x and y axis labels and a title to the plot
    ax.set_xlabel('Distribution')
    ax.set_ylabel('Fraction Occupancy')
    ax.set_title(' ')
    
    # Set the x-axis tick positions and labels
    ax.set_xticks(bar_offset)
    ax.set_xticklabels(pivot_df.index)
    
    # Set the y-axis limits
    ax.set_ylim(0, 1)
    # Set y-axis ticks
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    # Set x-ticks
    bar_offset = np.arange(len(pivot_df))
    ax.set_xticks(bar_offset + (1 * bar_width))
    ax.set_xticklabels([distribution_dict.get(x, x) for x in pivot_df.index])
    
    # Adjust the layout to avoid labels getting cut off
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def calculate_stim_dur(df):
    '''
    Function to calculate the average stimulus duration for each block for each participant
    Args:
        df (DataFrame): DataFrame containing the data
    Returns:
        result_df (DataFrame): DataFrame containing the average stimulus duration for each block for each participant
    '''
    result_df = pd.DataFrame(columns=['Participant', 'Block', 'Mean_Trial_Time', 'Mean_Stim_Duration'])
    
    for participant in df.Participant_ID.unique():
        df_Participant = df[df.Participant_ID == participant].copy()  # Make a copy to avoid modifying the original DataFrame
        for block in df_Participant.block.unique():
            block_df = df_Participant[df_Participant.block == block].copy()  # Make a copy
            no_response_df = block_df[block_df.No_response == True].copy()  # Make a copy
            no_response_df.loc[:, 'Time_Difference'] = no_response_df['Time'].diff()
            consec_trials_time_diffs = no_response_df[no_response_df['Trial'].diff() == 1]
            if not consec_trials_time_diffs.empty:
                mean_trial_time = consec_trials_time_diffs.Time_Difference.values.mean()
            else:
                mean_trial_time = np.nan
            if not np.isnan(mean_trial_time):
                mean_stim_dur = (mean_trial_time 
                                - pd.to_timedelta(block_df.GoCueDur).dt.total_seconds().values[0] 
                                - pd.to_timedelta(block_df.ITI).dt.total_seconds().values[0] 
                                - pd.to_timedelta(block_df.ReponseWindow).dt.total_seconds().values[0])
            else:
                mean_stim_dur = np.nan
            
            result_df = result_df.append({
                'Participant': participant,
                'Block': block,
                'Mean_Trial_Time': mean_trial_time,
                'Mean_Stim_Duration': mean_stim_dur
            }, ignore_index=True)
            result_df['True_Stim_Dur'] = result_df['Mean_Stim_Duration'].apply(lambda x: np.floor(x / 0.05) * 0.05)
            result_df['Estimated_True_Stim_Dur'] = result_df['Mean_Stim_Duration'].subtract(
                                                   result_df['Mean_Stim_Duration'].subtract(result_df['True_Stim_Dur']).mean()
                                                   )    
    return result_df

def plot_choice_scatter_V2(df, p_id, save_path, show_plot=True, plot_P_Right=False, jitter=0.025):
    participant_df = df[df['Participant_ID'] == p_id]
    participant_df = participant_df[participant_df['No_response'] == False]
    
    with PdfPages(save_path) as pdf:
        for block in participant_df['block'].unique():
            block_df = participant_df[participant_df['block'] == block]
            block_df = block_df[block_df['No_response'] == False].reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(18, 6))

            # Set the y-axis labels for primary axis
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Left', 'Right'])
            ax.set_ylim(-0.1, 1.1)  # Aligning the primary y-axis
            ax.set_xlabel('(Valid) Trial')
            ax.set_ylabel('Choice')

            # Create the secondary y-axis
            ax2 = ax.twinx()
            ax2.set_ylim(-0.1, 1.1)  # Aligning the secondary y-axis

            ax2.tick_params(axis='y', labelcolor='purple', which='both', left=False, right=True)  # Disable y-ticks on the right y-axis

            # Loop through the rows and plot based on choice and correctness
            for idx, row in block_df.iterrows():
                if row['choice'] == 0:
                    y_val = 0 + np.random.uniform(-jitter, jitter)
                else:  # Assuming 'Right' is the only other option
                    y_val = 1 + np.random.uniform(-jitter, jitter)

                if row['correct']:  # If the choice was correct
                    ax.plot(idx, y_val, 'o', color='green', markersize=7, alpha=0.5, label='Correct')
                else:  # If the choice was incorrect
                    ax.plot(idx, y_val, 'v', color='red', markersize=7, alpha=0.5, label='Incorrect')
                    
            if plot_P_Right:
                ax2.plot(block_df.index, block_df['P_Right'], color='purple', alpha=0.5, label='P(Right)')

            # Add a legend. Handles are the plotted elements, labels are the corresponding labels
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles, labels = handles1 + handles2, labels1 + labels2

            # To remove duplicate labels in the legend
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower left', bbox_to_anchor=(0, 0.1))

            plt.title(p_id + ' - ' + block_df['Datetime'].iloc[0].strftime('%Y/%m/%d') + ' - Block ' + str(block))
            plt.tight_layout()

            pdf.savefig()

            if show_plot:
                plt.show()
            else:
                plt.close()