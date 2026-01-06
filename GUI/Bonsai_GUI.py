# %%
# LIBRARIES

import os
import tkinter as tk
from tkinter import ttk
import csv
from tkinter import font
from tkinter import messagebox
import subprocess
import psutil 
import pandas as pd
import win32api
import win32file
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer, AsyncIOOSCUDPServer
import asyncio
import threading
import time
import sys
import numpy as np
import datetime
import shutil
import json
from pathlib import Path

# %%
# PATHS

repo_path = os.path.dirname('/'.join(os.getcwd().split('\\')))
protocols_path = repo_path + '/Protocols/' 
subject_params_file = repo_path + '/Params/Subject_Params.csv'
mouse_room_params_path = repo_path + '/Params/Mouse_Room_Params.xlsx'

user_profile  = ('/'.join(os.environ['USERPROFILE'].split('\\')))
bonsai_path = user_profile + '/AppData/Local/Bonsai/Bonsai.exe'

    
# %%
# add opto_type and zapit_nb_conditions to params

# FUNCTIONAL CODE

# Function to copy all contents
def copy_all_contents(src, dest, user=None, is_top_level=True):
    if is_top_level:
        if isinstance(user, str):
            user_str = user
        else:
            user_str = user.get()
    else:
        user_str = None  # Don't filter subdirectories
        
    if not os.path.exists(dest):
        os.makedirs(dest)

    for item in os.listdir(src):
        # Only filter at top level
        if is_top_level and user_str and not item.startswith(user_str):
            continue
            
        src_item = Path(src) / Path(item)
        dest_item = Path(dest) / Path(item)

        if os.path.isdir(src_item):
            if not os.path.exists(dest_item):
                shutil.copytree(src_item, dest_item)
            else:
                copy_all_contents(src_item, dest_item, user=user, is_top_level=False)
        else:
            if not os.path.exists(dest_item):
                shutil.copy2(src_item, dest_item)
def get_mapped_drives():
    drives = win32api.GetLogicalDriveStrings()
    drives = drives.split('\000')[:-1]
    mapped_drives = []
    for drive in drives:
        drive_type = win32file.GetDriveType(drive)
        if drive_type == win32file.DRIVE_REMOTE:
            mapped_drives.append(drive)
    return mapped_drives
    
def reset_overwrite_button(*args):
    overwrite_button.config(bg="orange")

def overwrite_csv():
    # Collect input values from user interface components
    params = [subject.get(), protocol.get(), stage.get(), distribution.get(), sound_duration.get(), nb_of_stim.get(),
              stim_type.get(), antibias.get(), emulator.get(), air_puff_contingency_rule.get(),
              show_contingency_switches.get(), working_memory_type.get(), sound_air_puff_contingency.get(),
              sound_contingency.get(), opto_on.get(), perc_opto_trials.get(), light_freq.get(), opto_onset_1.get(), opto_onset_2.get(),
              opto_offset_1.get(), opto_offset_2.get(), opto_duration.get(), arduino.get(), stimulation_site.get(), stimulation_type.get(),
              antibias_exp_rate.get(), antibias_window.get(), antibias_sigmoid_slope.get(), agent_sim.get(), agent_performance.get(),
              agent_bias.get(), stim_dur_staircase.get(), stim_dur_staircase_perf_thresh.get(), stim_dur_staircase_step.get(), min_stim_dur.get(),
              opto_type.get(), zapit_nb_conditions.get(), inter_trial_interval.get(), timeout_duration.get(), response_window.get(),
              stim_range_min.get(), stim_range_max.get(), go_cue_duration.get(), visualiser_window_size.get(), stable_start.get(), stable_start_window.get(),
              max_trials_consec.get(), stable_stim_dist_boundary.get()
              ,]

    # Check if any input values are "Select"
    if "Select" in params:
        messagebox.showwarning("Warning", "All params must be filled in")
        return

    # Join input values into a single string with commas and a trailing comma
    row = ", ".join([f"{name}: {value}" for name, value in zip(
                ['Subject', 'Protocol', 'Stage', 'Distribution', 'Sound_Duration', 'Nb_Of_Stim', 'Stim_Type', 
                 'AntiBias', 'Emulator', 'Air_Puff_Contingency_Rule', 'Show_Contingency_Switches', 'Working_Memory_Type', 
                 'Sound_Air_Puff_Contingency', 'Sound_Contingency', 'Opto_ON', 'Perc_Opto_Trials', 'Light_Freq (Hz)', 
                 'Opto_Onset_1', 'Opto_Onset_2', 'Opto_Offset_1', 'Opto_Offset_2', 'Opto_Duration', 'Arduino', 
                 'Stimulation_Site', 'Stimulation_Type', 'AntiBias_Exp_Rate', 'AntiBias_Window', 'AntiBias_Sigmoid_Slope',
                 'Agent_Sim', 'Agent_Performance', 'Agent_Bias', 'Stim_Dur_Staircase', 'Stim_Dur_Staircase_Perf_Thresh', 'Stim_Dur_Staircase_Step',
                 'Min_Stim_Dur', 'Opto_Type', 'Zapit_Nb_Conditions', 'Inter_Trial_Interval', 'Timeout_Duration', 'Response_Window',
                 'Stim_Range_Min', 'Stim_Range_Max', 'Go_Cue_Duration', 'Visualiser_Window_Size', 'Stable_Start', 'Stable_Start_Window',
                    'Max_Trials_Consec', 'Stable_Stim_Dist_Boundary'
                    ,],
                params)])

    # Add trailing comma to the end of the row
    row += ","

    # Write input values to CSV file
    with open(subject_params_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([row])

    # Display success message and change button colors 
    messagebox.showinfo("Success", "Overwrite successful")
    overwrite_button.config(bg='green')
    run_protocol_button.config(bg='green')
    
def load_csv():
    subj = subject.get()
    if subj == 'Select':
        tk.messagebox.showwarning("Warning", "Select a subject")
        return
    
    df = pd.read_excel(mouse_room_params_path, sheet_name = 'Params', converters={'opto_on': str})
    

    if subj in df['Subject'].unique():
        subj_params = df[df['Subject'] == subj]
        params = ['Protocol', 'Stage', 'Distribution', 'Sound_Duration', 'Nb_Of_Stim', 'Stim_Type', 
                 'AntiBias', 'Emulator', 'Air_Puff_Contingency_Rule', 'Show_Contingency_Switches', 'Working_Memory_Type', 
                 'Sound_Air_Puff_Contingency', 'Sound_Contingency', 'Opto_ON', 'Perc_Opto_Trials', 'Light_Freq (Hz)', 
                 'Opto_Onset_1', 'Opto_Onset_2', 'Opto_Offset_1', 'Opto_Offset_2', 'Opto_Duration', 'Arduino', 'Stimulation_Site', 'Stimulation_Type', 
                 'AntiBias_Exp_Rate', 'AntiBias_Window', 'AntiBias_Sigmoid_Slope', 'Agent_Sim', 'Agent_Performance', 'Agent_Bias', 'Stim_Dur_Staircase', 
                 'Stim_Dur_Staircase_Perf_Thresh', 'Stim_Dur_Staircase_Step', 'Min_Stim_Dur', 'Opto_Type', 'Zapit_Nb_Conditions', 'Inter_Trial_Interval', 
                 'Timeout_Duration', 'Response_Window', 'Stim_Range_Min', 'Stim_Range_Max', 'Go_Cue_Duration', 'Visualiser_Window_Size',
                 'Stable_Start', 'Stable_Start_Window', 'Max_Trials_Consec', 'Stable_Stim_Dist_Boundary',]
        vars_and_dropdowns = zip(params, [protocol, stage, distribution, sound_duration, nb_of_stim, stim_type, antibias, emulator,
                                          air_puff_contingency_rule, show_contingency_switches, working_memory_type, sound_air_puff_contingency,   
                                          sound_contingency, opto_on, perc_opto_trials, light_freq, opto_onset_1, opto_onset_2, opto_offset_1, opto_offset_2,
                                          opto_duration, arduino, stimulation_site, stimulation_type, antibias_exp_rate, 
                                          antibias_window, antibias_sigmoid_slope, agent_sim, agent_performance, agent_bias, stim_dur_staircase,
                                          stim_dur_staircase_perf_thresh, stim_dur_staircase_step, min_stim_dur, opto_type, zapit_nb_conditions,
                                          inter_trial_interval, timeout_duration, response_window, stim_range_min, stim_range_max, go_cue_duration, visualiser_window_size,
                                          stable_start, stable_start_window,
                                          max_trials_consec, stable_stim_dist_boundary
                                          ,],
                                 [protocol_dropdown, stage_dropdown, distribution_dropdown, sound_duration_dropdown,
                                  nb_of_stim_dropdown, stim_type_dropdown, antibias_dropdown, emulator_dropdown,
                                  air_puff_contingency_rule_dropdown, show_contingency_switches_dropdown, working_memory_type_dropdown,
                                  sound_air_puff_contingency_dropdown, sound_contingency_dropdown, opto_on_dropdown,
                                  perc_opto_trials_dropdown, light_freq_dropdown, opto_onset_1_dropdown, opto_onset_2_dropdown,
                                  opto_offset_1_dropdown, opto_offset_2_dropdown, opto_duration_dropdown, arduino_dropdown, stimulation_site_dropdown, 
                                  stimulation_type_dropdown, antibias_exp_rate_dropdown, antibias_window_dropdown, antibias_sigmoid_slope_dropdown,
                                  agent_sim_dropdown, agent_performance_dropdown, agent_bias_dropdown, stim_dur_staircase_dropdown,
                                  stim_dur_staircase_perf_thresh_dropdown, stim_dur_staircase_step_dropdown, min_stim_dur_dropdown, 
                                  opto_type_dropdown, zapit_nb_conditions_dropdown, inter_trial_interval_dropdown, timeout_duration_dropdown, response_window_dropdown,
                                  stim_range_min_dropdown, stim_range_max_dropdown, go_cue_duration_dropdown, visualiser_window_size_dropdown,
                                  stable_start_dropdown, stable_start_window_dropdown,
                                  max_trials_consec_dropdown, stable_stim_dist_boundary_dropdown
                                  ,])

        values = [subj] + [subj_params[param].values[0] for param in params]
        row = ", ".join([f"{name}: {value}" for name, value in zip(
                ['Subject'] + params, 
                values)])

        # Add trailing comma to the end of the row
        row += ","
        
        with open(subject_params_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([row,])

        for param, var, dropdown in vars_and_dropdowns:
            value = subj_params[param].values[0]
            if param in ['Stage', 'Nb_Of_Stim', 'AntiBias_Window', 'Zapit_Nb_Conditions', 'Stim_Range_Min', 
                         'Stim_Range_Max', 'Visualiser_Window_Size', 'Stable_Start_Window', 'Max_Trials_Consec']:
                try:
                    value = int(value)
                except:
                    pass
            elif param in ['Opto_ON', 'Speaker_Calib' , 'AntiBias', 'Emulator', 'Arduino', 
                           'Show_Contingency_Switches', 'Agent_Sim', 'Stim_Dur_Staircase', 'Stable_Start']:
                try:
                    value = str(value)
                except:
                    pass
            var.set(value)
            dropdown.config(bg='yellow')

        load_button.config(bg='green')
        tk.messagebox.showinfo("Success", "Params successfully loaded")
        run_protocol_button.config(bg='green')
    else:
        tk.messagebox.showwarning("Warning", "No params available for this subject")
        return
    
def launch_bonsai():
    global process
    if load_button['bg'] == 'orange' and overwrite_button['bg'] == 'orange':
        tk.messagebox.showwarning("Warning", "Protocol can't launch without params!")
        run_protocol_button.config(bg='orange')
    elif run_protocol_button['bg'] == 'green':
        file_path = protocols_path + 'Auditory_Discrimination/Sound_Cat_V2.bonsai'     
        
        if os.path.exists(file_path):
            process = subprocess.Popen([bonsai_path, file_path, '--start'])
            run_protocol_button.config(text='End', bg='crimson')
            overwrite_button.config(state='disabled')
            load_button.config(state='disabled')

        else:
            tk.messagebox.showwarning("Warning", "Protocol not found in on current machine")
            overwrite_button.config(bg='orange')
            load_button.config(bg='orange')
            run_protocol_button.config(bg='orange')

    elif run_protocol_button['bg'] == 'crimson':
        ip = "127.0.0.1"
        port = 1334
        client = SimpleUDPClient(ip,port)
        client.send_message("/GUI", "End_Protocol")
        run_protocol_button.config(text='Launch Bonsai', bg='orange')
        overwrite_button.config(state='active', bg='orange')
        load_button.config(state='active', bg='orange')


def kill_bonsai():
    kill_bonsai_button.config(bg='red')
    if process is not None:
        process.terminate()
        process.wait()
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == "Bonsai.exe":
                proc.kill()
                
def camera():
    if camera_button['bg']== 'green':
        camera_path =  repo_path + '/Params/Camera.bonsai'
        if os.path.exists(camera_path):
            global process
            process = subprocess.Popen([bonsai_path, camera_path, '--start'])
            camera_button.config(text='Camera', bg='crimson')
        else:
            tk.messagebox.showwarning("Warning", "No camera protocol found on current machine")
    elif camera_button['bg']== 'crimson':
        # call the kill_boonsai function
        kill_bonsai()
        camera_button.config(text='Camera', bg='green')
        
def flush_rig():
    if flush_rig_button['bg']== 'green':
        flush_rig_path =  repo_path + '/Params/Flush_Rig.bonsai'
        if os.path.exists(flush_rig_path):
            global process
            process = subprocess.Popen([bonsai_path, flush_rig_path, '--start'])
            flush_rig_button.config(bg='crimson')
        else:
            tk.messagebox.showwarning("Warning", "No flush rig protocol found on current machine")
    elif flush_rig_button['bg']== 'crimson':
        # call the kill_boonsai function
        kill_bonsai()
        flush_rig_button.config(bg='green')
        
def push_data():
    experimenter_str = experimenter.get()
    if experimenter_str == 'QP':
        CONFIG_FILE = repo_path +  "/GUI/paths/quentin.json"
    elif experimenter_str == 'SS':
        CONFIG_FILE = repo_path +  "/GUI/paths/serkan.json"
    else:
        print('Wrong Experimenter')
        

    with open(CONFIG_FILE, "r") as f:
        PATHS = json.load(f)
    
    for drive in get_mapped_drives():
        try:
            server_data_path = drive[0:2] + PATHS['data_path']
        except:
            pass
    local_data_path = repo_path + '/Data'
    if os.path.exists(server_data_path):
        copy_all_contents(local_data_path, server_data_path, experimenter)
    else:
        tk.messagebox.showwarning("Warning", "No server found on current machine")        
        
def create_label_dropdown(parent_frame, label_text, option_list, y_pos):
    var = tk.StringVar()
    var.set("Select")
    
    label = tk.Label(parent_frame, text=label_text, height=2, width=15, font=my_font)
    label.grid(row=y_pos, column=0, padx=10, pady=10)
    
    options = option_list
    dropdown = tk.OptionMenu(parent_frame, var, *options, 
                             command=lambda x: dropdown.config(bg="yellow"))
    dropdown.grid(row=y_pos, column=1, padx=10, pady=10)
    dropdown.config(height=1, width=16, font=my_font)
    
    return var, label, dropdown

# %%
# GUI CODE
        
root = tk.Tk()
root.title("Bonsai Launcher GUI")
root.geometry("600x775")
root.config(bg="gray")

# Create notebook
notebook = ttk.Notebook(root)
notebook.pack(pady=15, fill='both', expand=True)

# Style configuration
style = ttk.Style()
style.configure('TNotebook.Tab', font=('TkDefaultFont', 14), background = 'green')
my_font = font.Font(size=15)

# Create tabs
setup_tab = tk.Frame(notebook, bg='purple')
beh_tab_1 = tk.Frame(notebook, bg='purple')
beh_tab_2 = tk.Frame(notebook, bg='purple')
beh_tab_3 = tk.Frame(notebook, bg='purple')
beh_tab_4 = tk.Frame(notebook, bg='purple')
beh_tab_5 = tk.Frame(notebook, bg='purple')
stim_tab_1 = tk.Frame(notebook, bg='purple')
stim_tab_2 = tk.Frame(notebook, bg='purple')

notebook.add(setup_tab, text="Setup")
notebook.add(beh_tab_1, text="Beh_1")
notebook.add(beh_tab_2, text="Beh_2")
notebook.add(beh_tab_3, text="Beh_3")
notebook.add(beh_tab_4, text="Beh_4")
notebook.add(beh_tab_5, text="Beh_5")
notebook.add(stim_tab_1, text="Optostim_1")
notebook.add(stim_tab_2, text="Optostim_2")

# Add widgets to Tab 1
setup_frame = tk.Frame(setup_tab, bg='black')
setup_frame.pack(pady=30)

flush_rig_button = tk.Button(setup_frame, text="Flush Rig", bg='green',
                             height=1, width=10, font=my_font, command = flush_rig)
flush_rig_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

camera_button = tk.Button(setup_frame, text="Camera", bg='green',
                          height=1, width=10, font=my_font, command = camera)
camera_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

test_speakers_button = tk.Button(setup_frame, text="Test Speakers", state='disabled',
                                 height=1, width=15, font=my_font)
test_speakers_button.grid(row=1, column=1, padx=10, pady=10, sticky="w")

calibrate_button = tk.Button(setup_frame, text="Calibrate", state='disabled',
                             height=1, width=15, font=my_font)
calibrate_button.grid(row=2, column=1, padx=10, pady=10, sticky="w")

push_data_button = tk.Button(setup_frame, text="Push Data", bg='green',
                                height=1, width=10, font=my_font, command = push_data)
push_data_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")

experimenter, experimenter_label, experimenter_dropdown = create_label_dropdown(parent_frame = setup_frame, 
                                                                 label_text = "Experimenter:", 
                                                                 option_list = ['SS', 'QP'],
                                                                 y_pos = 0)

# Add widgets to Tab 2
beh_frame_1 = tk.Frame(beh_tab_1, bg='black')
beh_frame_1.pack(pady=30)


# Subject
# Get the current date

current_date = datetime.datetime.now()
# Get the current day of the week as a string
day_name = current_date.strftime('%A')

file_path = mouse_room_params_path
mouse_room_params_df = pd.read_excel(file_path, sheet_name='Params')

subject_option_list = mouse_room_params_df.Subject.unique().tolist() 
subject, subject_label, subject_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                 label_text = "Subject:", 
                                                                 option_list = subject_option_list,
                                                                 y_pos = 0)


# Protocol
protocol, protocol_label, protocol_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                    label_text = "Protocol:", 
                                                                    option_list = ["SOUND_CAT_DISC", "SOUND_CAT_CONT", 
                                                                                   "PRO_ANTI", "SOUND_CAT"], 
                                                                    y_pos = 1)

# Stage
stage, stage_label, stage_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                           label_text = "Stage:", 
                                                           option_list = ['Habituation', 'Lick_To_Release', 'Three_And_Three', 
                                                                          'Full_Task_Disc', 'Full_Task_Cont'], 
                                                           y_pos = 2)

# Rule
air_puff_contingency_rule, air_puff_contingency_rule_label, air_puff_contingency_rule_dropdown = create_label_dropdown(parent_frame = beh_frame_1,
                                                                                                                       label_text = "Rule:",
                                                                                                                       option_list = ['NaN', 'Pro_Only', 'Anti_Only', 
                                                                                                                                      'Blocks_30', 'Blocks_15', 'Random_Alternation'],
                                                                                                                       y_pos = 3)

# Anti_Bias
antibias, antibias_label, antibias_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                    label_text = "AntiBias:", 
                                                                    option_list = ['NaN', 'True', 'False'], 
                                                                    y_pos = 4)

# Distribution

distribution, distribution_label, distribution_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                                label_text = "Distribution:", 
                                                                                option_list =  ['NaN', 'Uniform', 
                                                                                                'Asym_Left', 'Asym_Right'], 
                                                                                y_pos = 5)
# nb_of_stim
nb_of_stim, nb_of_stim_label, nb_of_stim_dropdown = create_label_dropdown(parent_frame=beh_frame_1,
                                                                          label_text="Nb Of Stim:",
                                                                          option_list=[np.nan, 2, 4, 6, 8],
                                                                          y_pos=6)


load_button = tk.Button(beh_frame_1, text="Load params", command=load_csv, 
                        bg='orange', height=1, width=14, font=my_font)
load_button.grid(row=11, column=0, padx=10, pady=10)

overwrite_button = tk.Button(beh_frame_1, text="Overwrite params", command=overwrite_csv, 
                             bg='orange', height=1, width=14, font=my_font)
overwrite_button.grid(row=12, column=0, padx=10, pady=10)

run_protocol_button = tk.Button(beh_frame_1, text="Launch Bonsai", state='active', command=launch_bonsai, 
                                bg='orange', height=1, width=12, font=my_font)
run_protocol_button.grid(row=11, column=1, padx=10, pady=10)

kill_bonsai_button = tk.Button(beh_frame_1, text="Kill Bonsai", state='active', command=kill_bonsai, 
                               bg='red', height=1, width=12, font=my_font)
kill_bonsai_button.grid(row=12, column=1, padx=10, pady=10)

# Add widgets to Tab 2 (beh_tab_2)
beh_frame_2 = tk.Frame(beh_tab_2, bg='black')
beh_frame_2.pack(pady=30)

# Move dropdowns to beh_tab_2


# Stim_Type
stim_type, stim_type_label, stim_type_dropdown = create_label_dropdown(parent_frame = beh_frame_2, 
                                                                       label_text = "Stim Type:", 
                                                                       option_list = ['NaN', 'PT', 'WN'], 
                                                                       y_pos = 1)

emulator, emulator_label, emulator_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                    label_text="Emulator:",
                                                                    option_list=['True', 'False'],
                                                                    y_pos=2)

sound_duration, sound_duration_label, sound_duration_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                    label_text="Sound Duration:",
                                                                    option_list=[50 , 100, 150, 200, 250, 
                                                                                 300, 350, 400, 450, 500],
                                                                    y_pos=3)

arduino, arduino_label, arduino_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                 label_text="Arduino:",
                                                                 option_list=['NaN', 'True', 'False'],
                                                                 y_pos=4)

antibias_exp_rate, antibias_exp_rate_label, antibias_exp_rate_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                                               label_text="AB_Exp_Rate:",
                                                                                               option_list=[np.nan, 0.5, 1.0, 1.5, 
                                                                                                            2.0, 2.5, 3.0],
                                                                                               y_pos=5)

antibias_window, antibias_window_label, antibias_window_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                                         label_text="AB_Window:",
                                                                                         option_list=[np.nan, 10, 20, 30, 40, 50],
                                                                                         y_pos=6)

antibias_sigmoid_slope, antibias_sigmoid_slope_label, antibias_sigmoid_slope_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                                                              label_text="AB_Slope:",
                                                                                                              option_list=[np.nan, 0.5, 1.0, 1.5, 
                                                                                                                           2.0, 2.5, 3.0],
                                                                                                              y_pos=7)

# show contingency switches
show_contingency_switches, show_contingency_switches_label, show_contingency_switches_dropdown = create_label_dropdown(parent_frame = beh_frame_2,
                                                                                                                     label_text = "Show Contingency \n Switches:",  
                                                                                                                     option_list = ['NaN', 'True', 'False'],  
                                                                                                                     y_pos = 8)  

working_memory_type, working_memory_type_label, working_memory_type_dropdown = create_label_dropdown(parent_frame = beh_frame_2,
                                                                                                         label_text = "Working Memory \n Type:",
                                                                                                            option_list = ['NaN', 'Fixed', 'Variable'],
                                                                                                            y_pos = 9)

beh_frame_3 = tk.Frame(beh_tab_3, bg='black')
beh_frame_3.pack(pady=30)
                                                                                                
# working memory delay


# sound air puff contingency
sound_air_puff_contingency, sound_air_puff_contingency_label, sound_air_puff_contingency_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                                                                          label_text = "Sound Air  \n Puff Contingency:",
                                                                                                                            option_list = ['Low_Pro_High_Anti', 
                                                                                                                                           'Low_Anti_High_Pro'],
                                                                                                                            y_pos = 1)

# sound contingency
sound_contingency, sound_contingency_label, sound_contingency_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                                                    label_text = "Sound \n Contingency:",
                                                                                                    option_list = ['Low_Left_High_Right', 
                                                                                                                   'Low_Right_High_Left'],
                                                                                                    y_pos = 2)

# agent sim
agent_sim, agent_sim_label, agent_sim_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                        label_text = "Agent Sim:",
                                                                        option_list = ['NaN', 'True', 'False'],
                                                                        y_pos = 3)

# agent performance
agent_performance, agent_performance_label, agent_performance_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                                                label_text = "Agent \n Performance:",
                                                                                                option_list = ['NaN', '0.1', '0.2', '0.3', '0.4', '0.5',
                                                                                                                '0.6', '0.7', '0.8', '0.9', '1.0'],
                                                                                                y_pos = 4)

# agent bias
agent_bias, agent_bias_label, agent_bias_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                            label_text = "Agent Bias:",
                                                                            option_list = ['NaN', '0.1', '0.2', '0.3', '0.4', '0.5',
                                                                                             '0.6', '0.7', '0.8', '0.9', '1.0'],
                                                                            y_pos = 5)

# stim_dur_staircase
stim_dur_staircase, stim_dur_staircase_label, stim_dur_staircase_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                                                    label_text = "Stim Dur Staircase:",
                                                                                                    option_list = ['NaN', 'True', 'False'],
                                                                                                    y_pos = 6)

# stim_dur_staircase_perf_thresh
stim_dur_staircase_perf_thresh, stim_dur_staircase_perf_thresh_label, stim_dur_staircase_perf_thresh_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                                                                                label_text = "Stim Dur Staircase \n Perf Thresh:",
                                                                                                                                option_list = ['NaN', '0.1', '0.2', '0.3', '0.4', '0.5',
                                                                                                                                                  '0.6', '0.7', '0.8', '0.9', '1.0'],
                                                                                                                                y_pos = 7)

# stim_dur_staircase_step
stim_dur_staircase_step, stim_dur_staircase_step_label, stim_dur_staircase_step_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                                                                    label_text = "Stim Dur Staircase \n Step:",
                                                                                                                    option_list = ['NaN', '10', '20', '30', '40', '50'],
                                                                                                                    y_pos = 8)

# min_stim_dur
min_stim_dur, min_stim_dur_label, min_stim_dur_dropdown = create_label_dropdown(parent_frame = beh_frame_3,
                                                                                label_text = "Min Stim Dur:",
                                                                                option_list = ['NaN', '50', '100', '150',
                                                                                                '200', '250', '300'],
                                                                                y_pos = 9)


beh_frame_4 = tk.Frame(beh_tab_4, bg='black')
beh_frame_4.pack(pady=30)


# inter trial interval
inter_trial_interval, inter_trial_interval_label, inter_trial_interval_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                                        label_text = "Inter Trial \n Interval:",
                                                                                                        option_list = np.arange(0, 11, 1),
                                                                                                        y_pos = 1)

# timeout duration
timeout_duration, timeout_duration_label, timeout_duration_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                            label_text = "Timeout \n Duration:",
                                                                                            option_list = np.arange(0, 11, 1),
                                                                                            y_pos = 2)

# response window
response_window, response_window_label, response_window_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                            label_text = "Response \n Window:",
                                                                                            option_list = np.arange(0, 11, 1),
                                                                                            y_pos = 3)

# stim range min
stim_range_min, stim_range_min_label, stim_range_min_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                        label_text = "Stim Range \n Min:",
                                                                                        option_list = np.arange(40, 100, 1),
                                                                                        y_pos = 4)

# stim range max
stim_range_max, stim_range_max_label, stim_range_max_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                        label_text = "Stim Range \n Max:",
                                                                                        option_list = np.arange(40, 1000, 1),
                                                                                        y_pos = 5)

# go cur duration
go_cue_duration, go_cue_duration_label, go_cue_duration_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                            label_text = "Go Cue \n Duration:",
                                                                                            option_list = np.arange(40, 100, 100),
                                                                                            y_pos = 6)

# visualiser window size
visualiser_window_size, visualiser_window_size_label, visualiser_window_size_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                                            label_text = "Visualiser \n Window Size:",
                                                                                                            option_list = np.arange(10, 50, 5),
                                                                                                            y_pos = 7)

# stable start
stable_start, stable_start_label, stable_start_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                label_text = "Stable Start:",
                                                                                option_list = ['NaN', 'True', 'False'],
                                                                                y_pos = 8)

# stable start window
stable_start_window, stable_start_window_label, stable_start_window_dropdown = create_label_dropdown(parent_frame = beh_frame_4,
                                                                                                    label_text = "Stable Start \n Window:",
                                                                                                    option_list = np.arange(10, 55, 5),
                                                                                                    y_pos = 9)

beh_frame_5 = tk.Frame(beh_tab_5, bg='black')
beh_frame_5.pack(pady=30)     

# max trials consec
max_trials_consec, max_trials_consec_label, max_trials_consec_dropdown = create_label_dropdown(parent_frame = beh_frame_5,
                                                                                                label_text = "Max Trials \n Consec:",
                                                                                                option_list = np.arange(2, 11, 1),
                                                                                                y_pos = 1)

# stable stim dist boundary
stable_stim_dist_boundary, stable_stim_dist_boundary_label, stable_stim_dist_boundary_dropdown = create_label_dropdown(parent_frame = beh_frame_5,
                                                                                                                    label_text = "Stable Stim \n Dist Boundary:",
                                                                                                                    option_list = np.arange(0, 1, 0.1),
                                                                                                                    y_pos = 2)


# Add widgets to Tab 3
stim_frame_1 = tk.Frame(stim_tab_1, bg='black')
stim_frame_1.pack(pady=30)

# Opto_ON
opto_on, opto_on_label, opto_on_dropdown = create_label_dropdown(parent_frame = stim_frame_1, 
                                                                label_text = "Opto ON:", 
                                                                option_list = ['NaN', 'True', 'False'], 
                                                                y_pos = 0)

# Stim freq
light_freq, light_freq_label, light_freq_dropdown = create_label_dropdown(parent_frame = stim_frame_1, 
                                                                       label_text = "Light Freq (Hz):", 
                                                                       option_list = np.arange(0,110,10), 
                                                                       y_pos = 1)

# Perc opto trials
perc_opto_trials, perc_opto_trials_label, perc_opto_trials_dropdown = create_label_dropdown(parent_frame = stim_frame_1, 
                                                                                            label_text = "% Trials:", 
                                                                                            option_list = np.arange(0,110,5), 
                                                                                            y_pos = 2)

# Opto onset 1
opto_onset_1, opto_onset_label_1, opto_onset_1_dropdown = create_label_dropdown(parent_frame = stim_frame_1, 
                                                                                label_text = "Onset_1:", 
                                                                                option_list = ['Sound', 'Delay', 'Air_Puff', 'Go_Cue', 
                                                                                              'Response_Window', 'Feedback', 'Reward',
                                                                                              'Timeout', 'Inter_Trial_Interval'], 
                                                                                y_pos = 3)

# Opto onset 2
opto_onset_2, opto_onset_label_2, opto_onset_2_dropdown = create_label_dropdown(parent_frame = stim_frame_1,
                                                                                label_text = "Onset_2:",
                                                                                option_list = ['Sound', 'Delay', 'Air_Puff', 'Go_Cue',
                                                                                                'Response_Window', 'Feedback', 'Reward',
                                                                                                'Timeout', 'Inter_Trial_Interval'],
                                                                                y_pos = 4)

# Opto offset 1
opto_offset_1, opto_offset_label_1, opto_offset_1_dropdown = create_label_dropdown(parent_frame = stim_frame_1,
                                                                                    label_text = "Offset_1:",
                                                                                    option_list = ['Sound', 'Delay', 'Air_Puff', 'Go_Cue',
                                                                                                    'Response_Window', 'Feedback', 'Reward',
                                                                                                    'Timeout', 'Inter_Trial_Interval'],
                                                                                    y_pos = 5)

# Opto offset 2
opto_offset_2, opto_offset_label_2, opto_offset_2_dropdown = create_label_dropdown(parent_frame = stim_frame_1,
                                                                                    label_text = "Offset_2:",
                                                                                    option_list = ['Sound', 'Delay', 'Air_Puff', 'Go_Cue',
                                                                                                    'Response_Window', 'Feedback', 'Reward',
                                                                                                    'Timeout', 'Inter_Trial_Interval'],
                                                                                    y_pos = 6)

# Stimulation site
stimulation_site, stimulation_site_label, stimulation_site_dropdown = create_label_dropdown(parent_frame = stim_frame_1,
                                                                                            label_text = "Stim Site:",
                                                                                            option_list = ['NaN', 'PPC', 'ACC'],
                                                                                            y_pos = 7)

# Stimulation type
stimulation_type, stimulation_type_label, stimulation_type_dropdown = create_label_dropdown(parent_frame = stim_frame_1,
                                                                                            label_text = "Stim Type:",
                                                                                            option_list = ['NaN', 'Unilateral_Left', 
                                                                                                           'Unilateral_Right', 'Bilateral'],
                                                                                            y_pos = 8)
# # Opto duration
# opto_duration, opto_duration_label, opto_duration_dropdown = create_label_dropdown(parent_frame = stim_frame_1, 
#                                                                                    label_text = "Duration:", 
#                                                                                    option_list = np.arange(0, 1010, 100), 
#                                                                                    y_pos = 9) 

# make stim_frame_2
stim_frame_2 = tk.Frame(stim_tab_2, bg='black')
stim_frame_2.pack(pady=30)

# Opto duration
opto_duration, opto_duration_label, opto_duration_dropdown = create_label_dropdown(parent_frame = stim_frame_2, 
                                                                                   label_text = "Duration:", 
                                                                                   option_list = np.arange(0, 1010, 100), 
                                                                                   y_pos = 0) 

# Opto type
opto_type, opto_type_label, opto_type_dropdown = create_label_dropdown(parent_frame = stim_frame_2, 
                                                                       label_text = "Opto Type:", 
                                                                       option_list = ['NaN', 'Zapit', 'Fiber'], 
                                                                       y_pos = 1)

# Zapit nb conditions
zapit_nb_conditions, zapit_nb_conditions_label, zapit_nb_conditions_dropdown = create_label_dropdown(parent_frame = stim_frame_2, 
                                                                                                    label_text = "Zapit Nb \n Conditions:", 
                                                                                                    option_list = ['NaN', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                                                                                    y_pos = 2)

root.mainloop()


