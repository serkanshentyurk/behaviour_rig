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

# %%
# PATHS

repo_path = os.path.dirname('/'.join(os.getcwd().split('\\')))
protocols_path = repo_path + '/Protocols/' 
subject_params_file = repo_path + '/Params/Subject_Params.csv'
user_profile  = ('/'.join(os.environ['USERPROFILE'].split('\\')))
bonsai_path = user_profile + '/AppData/Local/Bonsai/Bonsai.exe'

# %%
# FUNCTIONAL CODE

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
    params = [subject.get(), stage.get(), protocol.get(), rule.get(), antibias.get(), distribution.get(), stim_type.get(), nb_of_stim.get(), 
              emulator.get(), speaker_calib.get(), stim_dur.get(), opto_on.get(), perc_opto_trials.get(), light_freq.get(), opto_onset.get(), 
              opto_offset.get(), opto_duration.get(), arduino.get(), stimulation_site.get(), stimulation_type.get(), antibias_exp_rate.get(), 
              antibias_window.get(), antibias_sigmoid_slope.get(),]

    # Check if any input values are "Select"
    if "Select" in params:
        messagebox.showwarning("Warning", "All params must be filled in")
        return

    # Join input values into a single string with commas and a trailing comma
    row = ", ".join([f"{name}: {value}" for name, value in zip(
                ["Subject", "Stage", "Protocol", "Rule", "AntiBias", "Distribution", "Stim_Type", "Nb_Of_Stim", 
                 "Emulator", "Speaker_Calib", "Stim_Dur", "Opto_ON", "Perc_Opto_Trials", "Light_Freq (Hz)", 
                 "Opto_Onset", "Opto_Offset", "Opto_Duration", "Arduino", "Stimulation_Site", "Stimulation_Type", 
                 "Anti_Bias_Exp_Rate", "Anti_Bias_Window", "Anti_Bias_Sigmoid_Slope",], 
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

    for drive in get_mapped_drives():
        try:
            df = pd.read_excel(drive[0:2] + '/Quentin/Head_Fixed_Behavior/Params/Mouse_Room_Params.xlsx', sheet_name='Sheet1', converters={'opto_on': str})
        except:
            pass

    if subj in df['Subject'].unique():
        subj_params = df[df['Subject'] == subj]
        params = ['Protocol', 'Stage', 'Rule', 'AntiBias', 'Distribution', 'Stim_Type', 'Nb_Of_Stim', 'Emulator', 
                  'Speaker_Calib', 'Stim_Dur', 'Opto_ON', 'Perc_Opto_Trials', 'Light_Freq (Hz)', 'Opto_Onset', 
                  'Opto_Offset', 'Opto_Duration', 'Arduino', 'Stimulation_Site', 'Stimulation_Type', 'Anti_Bias_Exp_Rate', 
                  'Anti_Bias_Window', 'Anti_Bias_Sigmoid_Slope',]
        vars_and_dropdowns = zip(params, [protocol, stage, rule, antibias, distribution, stim_type, 
                                          nb_of_stim, emulator, speaker_calib, stim_dur, opto_on, 
                                          perc_opto_trials, light_freq, opto_onset, opto_offset, 
                                          opto_duration, arduino, stimulation_site, stimulation_type,
                                          antibias_exp_rate, antibias_window, antibias_sigmoid_slope,],
                                 [protocol_dropdown, stage_dropdown, rule_dropdown, antibias_dropdown, 
                                  distribution_dropdown, stim_type_dropdown, nb_of_stim_dropdown, 
                                  emulator_dropdown, speaker_calib_dropdown, stim_dur_dropdown, 
                                  opto_on_dropdown, perc_opto_trials_dropdown, light_freq_dropdown, 
                                  opto_onset_dropdown, opto_offset_dropdown, opto_duration_dropdown, 
                                  arduino_dropdown, stimulation_site_dropdown, stimulation_type_dropdown,
                                  antibias_exp_rate_dropdown, antibias_window_dropdown, antibias_sigmoid_slope_dropdown,])

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
            if param in ['Stage', 'Nb_Of_Stim']:
                try:
                    value = int(value)
                except:
                    pass
            elif param in ['Opto_ON', 'Speaker_Calib' , 'AntiBias', 'Emulator', 'Arduino']:
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
        if protocol.get() == 'SOUND_CAT_DISC':
            file_path = protocols_path + 'Auditory_Discrimination/Sound_Cat_Disc.bonsai'     
        elif protocol.get() == 'SOUND_CAT_CONT':
            file_path = protocols_path + 'Auditory_Discrimination/Sound_Cat_Cont.bonsai'
        elif protocol.get() == 'PRO_ANTI':
            file_path = protocols_path + 'Pro_Anti/Pro_Anti.bonsai'
        else:
            file_path = ''
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
        
        
def create_label_dropdown(parent_frame, label_text, option_list, y_pos):
    var = tk.StringVar()
    var.set("Select")
    
    label = tk.Label(parent_frame, text=label_text, height=1, width=12, font=my_font)
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
root.geometry("450x675")
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
stim_tab = tk.Frame(notebook, bg='purple')

notebook.add(setup_tab, text="Setup")
notebook.add(beh_tab_1, text="Behaviour_1")
notebook.add(beh_tab_2, text="Behaviour_2")
notebook.add(stim_tab, text="Optostim")

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

# Add widgets to Tab 2
beh_frame_1 = tk.Frame(beh_tab_1, bg='black')
beh_frame_1.pack(pady=30)


# Subject
# Get the current date

current_date = datetime.datetime.now()
# Get the current day of the week as a string
day_name = current_date.strftime('%A')
for drive in get_mapped_drives():
    try:
        file_path = drive[0:2] + '/Quentin/Head_Fixed_Behavior/Params/Mouse_Room_Params.xlsx'
        mouse_room_params_df = pd.read_excel(file_path, sheet_name='Sheet1')
    except:
        pass

subject_option_list = mouse_room_params_df.Subject.unique().tolist() 
subject, subject_label, subject_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                 label_text = "Subject:", 
                                                                 option_list = subject_option_list,
                                                                 y_pos = 0)


# Protocol
protocol, protocol_label, protocol_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                    label_text = "Protocol:", 
                                                                    option_list = ["SOUND_CAT_DISC", "SOUND_CAT_CONT", 
                                                                                 "SOUND_CAT_DISC_V2", "SOUND_CAT_CONT_V2",
                                                                                 "PRO_ANTI"], 
                                                                    y_pos = 1)

# Stage
stage, stage_label, stage_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                           label_text = "Stage:", 
                                                           option_list = [np.nan, 1, 2, 3, 4], 
                                                           y_pos = 2)

# Rule
rule, rule_label, rule_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                        label_text = "Rule:", 
                                                        option_list = ['NaN', 'Pro_Only','Anti_Only', 'Blocks_30', 'Blocks_15', 'Random_Alternation'], 
                                                        y_pos = 3)

# Anti_Bias
antibias, antibias_label, antibias_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                    label_text = "AntiBias:", 
                                                                    option_list = ['NaN', 'True', 'False'], 
                                                                    y_pos = 4)

# Distribution

distribution, distribution_label, distribution_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                                label_text = "Distribution:", 
                                                                                option_list =  ['NaN', 'Uniform', 'Asym_Left', 'Asym_Right'], 
                                                                                y_pos = 5)
# Stim_Type
stim_type, stim_type_label, stim_type_dropdown = create_label_dropdown(parent_frame = beh_frame_1, 
                                                                       label_text = "Stim Type:", 
                                                                       option_list = ['NaN', 'PT', 'WN'], 
                                                                       y_pos = 6)


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
nb_of_stim, nb_of_stim_label, nb_of_stim_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                          label_text="Nb Of Stim:",
                                                                          option_list=[np.nan, 2, 4, 6, 8],
                                                                          y_pos=1)

emulator, emulator_label, emulator_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                    label_text="Emulator:",
                                                                    option_list=['True', 'False'],
                                                                    y_pos=2)

speaker_calib, speaker_calib_label, speaker_calib_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                                   label_text="Speaker Calib:",
                                                                                   option_list=['NaN', 'True', 'False'],
                                                                                   y_pos=3)

stim_dur, stim_dur_label, stim_dur_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                    label_text="Stim Duration:",
                                                                    option_list=[100, 200, 300, 400, 500],
                                                                    y_pos=4)

arduino, arduino_label, arduino_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                 label_text="Arduino:",
                                                                 option_list=['NaN', 'True', 'False'],
                                                                 y_pos=5)

antibias_exp_rate, antibias_exp_rate_label, antibias_exp_rate_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                                               label_text="AB_Exp_Rate:",
                                                                                               option_list=[np.nan, 0.5, 1.0, 1.5, 
                                                                                                            2.0, 2.5, 3.0],
                                                                                               y_pos=6)

antibias_window, antibias_window_label, antibias_window_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                                         label_text="AB_Window:",
                                                                                         option_list=[np.nan, 10, 20, 30, 40, 50],
                                                                                         y_pos=7)

antibias_sigmoid_slope, antibias_sigmoid_slope_label, antibias_sigmoid_slope_dropdown = create_label_dropdown(parent_frame=beh_frame_2,
                                                                                                              label_text="AB_Slope:",
                                                                                                              option_list=[np.nan, 0.5, 1.0, 1.5, 
                                                                                                                           2.0, 2.5, 3.0],
                                                                                                              y_pos=8)

# Add widgets to Tab 3
stim_frame = tk.Frame(stim_tab, bg='black')
stim_frame.pack(pady=30)

# Opto_ON
opto_on, opto_on_label, opto_on_dropdown = create_label_dropdown(parent_frame = stim_frame, 
                                                                label_text = "Opto ON:", 
                                                                option_list = ['NaN', 'True', 'False'], 
                                                                y_pos = 0)

# Stim freq
light_freq, light_freq_label, light_freq_dropdown = create_label_dropdown(parent_frame = stim_frame, 
                                                                       label_text = "Light Freq (Hz):", 
                                                                       option_list = np.arange(0,110,10), 
                                                                       y_pos = 1)

# Perc opto trials
perc_opto_trials, perc_opto_trials_label, perc_opto_trials_dropdown = create_label_dropdown(parent_frame = stim_frame, 
                                                                                            label_text = "% Trials:", 
                                                                                            option_list = np.arange(0,110,5), 
                                                                                            y_pos = 2)

# Opto onset
opto_onset, opto_onset_label, opto_onset_dropdown = create_label_dropdown(parent_frame = stim_frame, 
                                                                          label_text = "Onset:", 
                                                                          option_list = ['Stimulus', 'Go_Cue', 'Response', 'Feedback', 'ITI'], 
                                                                          y_pos = 3)

# Opto offset
opto_offset, opto_offset_label, opto_offset_dropdown = create_label_dropdown(parent_frame = stim_frame, 
                                                                             label_text = "Offset:", 
                                                                             option_list = ['Stimulus', 'Go_Cue', 'Response', 'Feedback', 'ITI'], 
                                                                             y_pos = 4)

# Opto duration
opto_duration, opto_duration_label, opto_duration_dropdown = create_label_dropdown(parent_frame = stim_frame, 
                                                                                   label_text = "Duration:", 
                                                                                   option_list = np.arange(0,1010,100), 
                                                                                   y_pos = 5)

# Stimulation site
stimulation_site, stimulation_site_label, stimulation_site_dropdown = create_label_dropdown(parent_frame = stim_frame,
                                                                                            label_text = "Stim Site:",
                                                                                            option_list = ['NaN', 'PPC', 'ACC'],
                                                                                            y_pos = 6)

# Stimulation type
stimulation_type, stimulation_type_label, stimulation_type_dropdown = create_label_dropdown(parent_frame = stim_frame,
                                                                                            label_text = "Stim Type:",
                                                                                            option_list = ['NaN', 'Unilateral_Left', 
                                                                                                           'Unilateral_Right', 'Bilateral'],
                                                                                            y_pos = 7)


root.mainloop()


