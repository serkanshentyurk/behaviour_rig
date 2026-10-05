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

import params_io

# %%
# PATHS

repo_path              = str(Path(__file__).resolve().parent.parent)
gui_dir                = str(Path(__file__).resolve().parent)
protocols_path         = repo_path + '/Protocols/'
subject_params_file    = repo_path + '/Params/Subject_Params.csv'
mouse_room_params_path = repo_path + '/Params/Mouse_Room_Params.xlsx'

user_profile = str(Path.home())
bonsai_path  = user_profile + '/AppData/Local/Bonsai/Bonsai.exe'

process = None   # set by launch_bonsai/camera/flush_rig; kill_bonsai reads it


# %%
# RIG IDENTITY
# Each machine carries C:/ProgramData/MouseRoom/rig.json = {"rig": "<id>"}.
# The GUI reads it, finds that rig's row in Params/Rigs.csv (the committed source
# of truth), and writes Params/Rig_Params.csv in the flat "Key: value," format the
# Bonsai workflow parses. This replaces the old per-machine Desktop copy: rig
# calibration now lives in version control, one row per machine.

RIG_JSON  = r"C:\ProgramData\MouseRoom\rig.json"
RIGS_CSV  = repo_path + "/Params/Rigs.csv"
rig_params_file = repo_path + "/Params/Rig_Params.csv"

# The columns written to Rig_Params.csv (params_io.RIG_PARAM_COLS) include Arduino,
# normalised from Rigs.csv's TRUE/FALSE to the True/False the workflow compares with.


def resolve_rig():
    """Read rig.json, look up the row in Rigs.csv, write Rig_Params.csv.
    Returns the rig row (dict). Raises with a clear message on any failure -
    the caller aborts launch rather than running on the wrong calibration."""
    import json, csv, socket

    if not os.path.exists(RIG_JSON):
        raise RuntimeError("No rig.json on this machine (expected at %s).\n"
                           "This machine has not been registered as a rig." % RIG_JSON)
    with open(RIG_JSON) as f:
        rig_id = json.load(f)['rig']

    with open(RIGS_CSV) as f:
        rows = {r['rig']: r for r in csv.DictReader(f)}
    if rig_id not in rows:
        raise RuntimeError("rig.json says '%s' but that is not a row in Rigs.csv.\n"
                           "Known rigs: %s" % (rig_id, ", ".join(sorted(rows))))
    row = rows[rig_id]

    # Cross-check hostname: catches a cloned disk image where rig.json rode along.
    host = socket.gethostname()
    if row.get('Hostname') and row['Hostname'] != host:
        raise RuntimeError("rig.json says '%s', whose Hostname is '%s' in Rigs.csv,\n"
                           "but this machine is '%s'. Two machines sharing a rig id?"
                           % (rig_id, row['Hostname'], host))

    pairs = params_io.rig_pairs_from_row(row)
    problems = params_io.check_pairs(pairs, params_io.RIG_BOOL_KEYS)
    if problems:
        raise RuntimeError("Rigs.csv row '%s' is not usable:\n%s" % (rig_id, "\n".join(problems)))
    params_io.write_rig_params(rig_params_file, pairs)
    return row


    
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
    return [p.device for p in psutil.disk_partitions(all=True) if 'remote' in p.opts]


# ---- application state ----------------------------------------------------
# Explicit state, NOT button colours. The old code read run_protocol_button['bg']
# to decide what to do, which meant the buttons could never be restyled - and on
# macOS the Aqua theme ignores Button bg outright, so the state was invisible.
import platform
IS_MAC = platform.system() == 'Darwin'

# ---- palette (muted, coherent; button colours still encode state) ----------
COL_OK      = '#4c956c'   # go / loaded / ready        (muted green)
COL_WARN    = '#e8a44c'   # needs attention / unsaved  (amber)
COL_STOP    = '#c44536'   # running / kill             (brick red)
COL_IDLE    = '#9aa0a6'   # disabled / not-ready       (grey)
COL_DIRTY   = '#f2cc55'   # a field just changed       (soft yellow)
COL_ACCENT  = '#2f3e46'   # headers / selected tab text
COL_MUTE    = '#8a9199'   # secondary text
# Backgrounds: the rigs run light mode, so give them an intentional off-white
# card-on-canvas look. On macOS leave them as None so the app follows the
# system theme instead of clashing with a dark title bar.
if IS_MAC:
    CANVAS = None         # window / tab strip
    CARD   = None         # the content panel
else:
    CANVAS = '#eef1f4'    # soft cool grey
    CARD   = '#ffffff'    # white card the fields sit on
FONT_UI     = ('Segoe UI', 12)
FONT_LABEL  = ('Segoe UI', 12)
FONT_BTN    = ('Segoe UI', 12, 'bold')
FONT_TAB    = ('Segoe UI', 11)
FONT_RIG    = ('Segoe UI', 12, 'bold')


class State:
    loaded  = False   # params have been Loaded or Overwritten at least once
    dirty   = False   # a widget changed since then: the CSV no longer matches
    running   = False   # Bonsai is up
    camera_on = False
    flush_on  = False
    ended_at  = None    # time.time() when End_Protocol was last sent

S = State()


def _shade(hex_colour, factor=0.88):
    """Darken a #rrggbb colour for the active/pressed state."""
    try:
        r = int(hex_colour[1:3], 16); g = int(hex_colour[3:5], 16); b = int(hex_colour[5:7], 16)
        return '#%02x%02x%02x' % (int(r*factor), int(g*factor), int(b*factor))
    except Exception:
        return hex_colour

def _paint(btn, colour):
    """macOS Aqua ignores Button bg; highlightbackground is the one that tints."""
    if IS_MAC:
        btn.config(highlightbackground=colour, activebackground=colour, fg='#ffffff')
    else:
        btn.config(bg=colour, activebackground=_shade(colour), fg='#ffffff',
                   disabledforeground='#eef1f4')


def refresh_buttons():
    """Single place where state -> appearance. Nothing else touches colours."""
    if S.running:
        _paint(run_protocol_button, COL_STOP)
        run_protocol_button.config(text='End Session')
        load_button.config(state='disabled')
        overwrite_button.config(state='disabled')
    else:
        run_protocol_button.config(text='Launch Bonsai', state='active')
        load_button.config(state='active')
        overwrite_button.config(state='active')
        ready = S.loaded and not S.dirty
        _paint(run_protocol_button, COL_OK if ready else COL_IDLE)

    _paint(load_button,      COL_OK if S.loaded else COL_WARN)
    _paint(overwrite_button, COL_WARN if (S.dirty or not S.loaded) else COL_OK)

    status_label.config(
        text = 'Bonsai running'                    if S.running   else
               'Unsaved changes - press Overwrite' if S.dirty     else
               'Ready to launch'                   if S.loaded    else
               'Load or Overwrite params to begin')


def mark_dirty(*args):
    S.dirty = True
    refresh_buttons()


def get_params():
    """View over SPEC + WIDGETS.
    (key written to Subject_Params.csv, tk variable, xlsx column, widget, cast)"""
    return [(key, WIDGETS[key][0], col, WIDGETS[key][2], cast)
            for key, col, _, _, cast, _ in SPEC]

def overwrite_csv():
    # Collect input values from user interface components
    rows = get_params()
    params = [var.get() for _, var, _, _, _ in rows]

    # Check if any input values are "Select"
    if "Select" in params:
        messagebox.showwarning("Warning", "All params must be filled in")
        return

    pairs = [(key, value) for (key, _, _, _, _), value in zip(rows, params)]
    problems = params_io.check_pairs(pairs, params_io.BOOL_KEYS)
    if problems:
        messagebox.showwarning("Warning", "Not saved:\n" + "\n".join(problems))
        return

    params_io.write_subject_params(subject_params_file, pairs)

    # Display success message and change button colors 
    S.loaded = True
    S.dirty  = False
    refresh_buttons()
    messagebox.showinfo("Success", "Overwrite successful")
    
def load_csv():
    subj = WIDGETS['Animal_ID'][0].get()
    if subj == 'Select':
        tk.messagebox.showwarning("Warning", "Select a subject")
        return
    
    df = pd.read_excel(mouse_room_params_path, sheet_name = 'Params', converters={'opto_on': str})
    

    if subj in df['Subject'].unique():
        subj_params = df[df['Subject'] == subj]
        rows = get_params()
        params = [col for _, _, col, _, _ in rows[1:]]
        missing = [col for col in params if col not in df.columns]
        if missing:
            tk.messagebox.showwarning("Warning", "Spreadsheet is missing columns:\n" + "\n".join(missing))
            return
        vars_and_dropdowns = zip(params,
                                 [var  for _, var, _, _, _  in rows[1:]],
                                 [dd   for _, _, _, dd, _   in rows[1:]],
                                 [cast for _, _, _, _, cast in rows[1:]])

        pairs = params_io.subject_pairs_from_row(subj_params.iloc[0])
        problems = params_io.check_pairs(pairs, params_io.BOOL_KEYS)
        if problems:
            tk.messagebox.showwarning("Warning", "Not loaded - fix the spreadsheet row:\n" + "\n".join(problems))
            return

        params_io.write_subject_params(subject_params_file, pairs)

        for param, var, dropdown, cast in vars_and_dropdowns:
            value = subj_params[param].values[0]
            if cast is not None:
                try:
                    value = cast(value)
                except Exception:
                    pass
            var.set(value)
            _paint(dropdown, COL_DIRTY)

        S.loaded = True
        S.dirty  = False
        refresh_buttons()
        tk.messagebox.showinfo("Success", "Params successfully loaded")
    else:
        tk.messagebox.showwarning("Warning", "No params available for this subject")
        return
    
def launch_bonsai():
    global process
    if S.running:
        send_end_protocol()
        S.running = False
        refresh_buttons()
        return

    if not S.loaded:
        tk.messagebox.showwarning("Warning", "Protocol can't launch without params!")
        return
    if S.dirty:
        tk.messagebox.showwarning("Warning",
                                  "Params changed since last save - press Overwrite first")
        return

    file_path = protocols_path + 'Auditory_discrimination/Sound_Cat_V2.bonsai'
    if not os.path.exists(file_path):
        tk.messagebox.showwarning("Warning", "Protocol not found on current machine")
        return
    if not os.path.exists(bonsai_path):
        tk.messagebox.showwarning("Warning", "Bonsai.exe not found at " + bonsai_path)
        return

    process = subprocess.Popen([bonsai_path, file_path, '--start'], cwd=gui_dir)
    S.running = True
    refresh_buttons()


CLOSE_FILES_WAIT_S = 4   # the workflow closes its CSVs 2 s after End_Protocol


def send_end_protocol():
    SimpleUDPClient("127.0.0.1", 1334).send_message("/GUI", "End_Protocol")
    S.ended_at = time.time()


def kill_bonsai():
    """Stop Bonsai without losing the end of the data files.

    Killing the process discards whatever the CsvWriters still hold in memory, so
    if a session is running, End_Protocol goes out first and the kill waits until
    the workflow has had time to close its files."""
    if S.running:
        send_end_protocol()
        S.running = False
        refresh_buttons()
    wait_s = 0 if S.ended_at is None else CLOSE_FILES_WAIT_S - (time.time() - S.ended_at)
    if wait_s > 0:
        status_label.config(text='Closing data files...')
        root.after(int(wait_s * 1000), _terminate_bonsai)
    else:
        _terminate_bonsai()


def _terminate_bonsai():
    """The original kill: only acts if this GUI launched something."""
    if process is not None:
        process.terminate()
        process.wait()
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == "Bonsai.exe":
                proc.kill()
    S.ended_at = None
    refresh_buttons()
                
def camera():
    global process
    if S.camera_on:
        _terminate_bonsai()
        S.camera_on = False
        _paint(camera_button, COL_OK)
        return
    camera_path = repo_path + '/Params/Camera.bonsai'
    if not os.path.exists(camera_path):
        tk.messagebox.showwarning("Warning", "No camera protocol found on current machine")
        return
    process = subprocess.Popen([bonsai_path, camera_path, '--start'], cwd=gui_dir)
    S.camera_on = True
    _paint(camera_button, COL_STOP)

def open_in_editor():
    """Open the workflow in the Bonsai editor with the SAME working directory a
    real session uses. Opening Bonsai from the Start menu gives CWD =
    AppData\\Local\\Bonsai, where neither ..\\Params\\ nor Extensions\\ resolves."""
    file_path = protocols_path + 'Auditory_discrimination/Sound_Cat_V2.bonsai'
    if not os.path.exists(file_path):
        tk.messagebox.showwarning("Warning", "Protocol not found on current machine")
        return
    if not os.path.exists(bonsai_path):
        tk.messagebox.showwarning("Warning", "Bonsai.exe not found at " + bonsai_path)
        return
    subprocess.Popen([bonsai_path, file_path], cwd=gui_dir)      # no --start
    
def flush_rig():
    global process
    if S.flush_on:
        _terminate_bonsai()
        S.flush_on = False
        _paint(flush_rig_button, COL_OK)
        return
    flush_rig_path = repo_path + '/Params/Flush_Rig.bonsai'
    if not os.path.exists(flush_rig_path):
        tk.messagebox.showwarning("Warning", "No flush rig protocol found on current machine")
        return
    process = subprocess.Popen([bonsai_path, flush_rig_path, '--start'], cwd=gui_dir)
    S.flush_on = True
    _paint(flush_rig_button, COL_STOP)


def push_data():
    experimenter_str = experimenter.get()
    CONFIG_FILE = repo_path + "/GUI/paths/%s.json" % {'QP': 'quentin', 'SS': 'serkan'}.get(experimenter_str, '')

    if experimenter_str not in ('QP', 'SS'):
        tk.messagebox.showwarning("Warning", "Select an experimenter before pushing")
        return

    with open(CONFIG_FILE, "r") as f:
        PATHS = json.load(f)

    server_data_path = None
    for drive in get_mapped_drives():
        candidate = drive[0:2] + PATHS['data_path']
        if os.path.exists(candidate):
            server_data_path = candidate
            break

    if server_data_path is None:
        tk.messagebox.showwarning("Warning", "No server found on current machine")
        return

    copy_all_contents(repo_path + '/Data', server_data_path, experimenter)      
        
def create_label_dropdown(parent_frame, label_text, option_list, y_pos):
    var = tk.StringVar()
    var.set("Select")

    _lbl_bg = {'bg': CARD} if CARD else {}
    label = tk.Label(parent_frame, text=label_text.replace('\n', ''), width=24,
                     font=FONT_LABEL, fg=COL_ACCENT, anchor='e', justify='right', **_lbl_bg)
    label.grid(row=y_pos, column=0, padx=(10, 14), pady=5, sticky='e')

    dropdown = tk.OptionMenu(parent_frame, var, *option_list,
                             command=lambda x: (_paint(dropdown, COL_DIRTY), mark_dirty()))
    dropdown.grid(row=y_pos, column=1, padx=10, pady=5, sticky='w')
    dropdown.config(height=1, width=16, font=FONT_UI, relief='flat',
                    bg='#f4f6f7', activebackground='#e4e7ea',
                    highlightthickness=1, highlightbackground='#c8ccd0',
                    borderwidth=0)

    return var, label, dropdown


# %%
# GUI CODE

# ---- window ---------------------------------------------------------------
# No hardcoded background colours. tk resolves the defaults to the system
# window colour, which follows light/dark mode on macOS and matches the shell
# on Windows. Hardcoding a light palette breaks dark mode; hardcoding a dark
# one breaks the rigs.
#
# Button colours ARE hardcoded and must stay: launch_bonsai reads
# run_protocol_button['bg'] to decide what to do, so 'orange' / 'green' /
# 'crimson' are program state, not styling. NB on macOS the Aqua theme ignores
# Button bg entirely, so they render grey there. The logic still works - you
# just can't see the state. On Windows they colour normally.

root = tk.Tk()
root.title("Bonsai Launcher GUI")
if CANVAS: root.config(bg=CANVAS)
root.geometry("880x720")
root.minsize(860, 620)

# The action bar packs FIRST, anchored bottom, so it reserves its strip before
# the notebook expands. No fill='x' - a shrink-to-fit frame is centred by pack.
action_frame = tk.Frame(root, bg=CANVAS) if CANVAS else tk.Frame(root)
action_frame.pack(side='bottom', pady=10)

notebook = ttk.Notebook(root)
notebook.pack(side='top', pady=10, padx=10, fill='both', expand=True)

style = ttk.Style()
try:
    style.theme_use('clam')
except Exception:
    pass

_canvas = CANVAS or '#eef1f4'
_card   = CARD   or '#ffffff'
style.configure('TNotebook', background=_canvas, borderwidth=0, tabmargins=[10, 8, 10, 0])
style.configure('TNotebook.Tab', font=FONT_TAB, padding=[16, 8], borderwidth=0,
                background='#dfe3e8', foreground=COL_MUTE)
style.map('TNotebook.Tab',
          background=[('selected', _card)],
          foreground=[('selected', COL_ACCENT)],
          expand=[('selected', [0, 0, 0, 0])])
# soft rounded-ish frame around content (clam draws a thin flat border, not a bevel)
style.configure('Card.TFrame', background=_card)
my_font = font.Font(family='Segoe UI', size=12)

# Subject list comes from the spreadsheet; SPEC references it, so it must exist first.
mouse_room_params_df = pd.read_excel(mouse_room_params_path, sheet_name='Params')
subject_option_list  = mouse_room_params_df.Subject.unique().tolist()

# SPEC (the single source of truth for keys, xlsx columns, options and tabs)
# lives in params_io so it can be tested without opening the window.
TABS = params_io.TABS
SPEC = params_io.build_spec(subject_option_list)


# %%
# BUILD THE TABS

tabs   = {}
frames = {}
for t in TABS:
    tabs[t] = tk.Frame(notebook, bg=CARD) if CARD else tk.Frame(notebook)
    notebook.add(tabs[t], text=t)
    frames[t] = tk.Frame(tabs[t], bg=CARD) if CARD else tk.Frame(tabs[t])
    frames[t].pack(pady=(24, 0), anchor='n')

setup_frame = frames['Setup']

WIDGETS = {}
for t in TABS:
    for i, (key, col, label, options, cast, tab) in enumerate([s for s in SPEC if s[5] == t]):
        WIDGETS[key] = create_label_dropdown(frames[t], label, options, y_pos=i)

# %%
# SETUP TAB

experimenter, experimenter_label, experimenter_dropdown = create_label_dropdown(
    parent_frame=setup_frame, label_text="Experimenter:",
    option_list=['SS', 'QP'], y_pos=0)

flush_rig_button = tk.Button(setup_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=8, pady=6,
                             text="Flush Rig", width=12, command=flush_rig)
flush_rig_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

camera_button = tk.Button(setup_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=8, pady=6,
                          text="Camera", width=12, command=camera)
camera_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

test_speakers_button = tk.Button(setup_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=8, pady=6, text="Test Speakers", state='disabled', width=15)
test_speakers_button.grid(row=1, column=1, padx=10, pady=10, sticky="w")

calibrate_button = tk.Button(setup_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=8, pady=6, text="Calibrate", state='disabled', width=15)
calibrate_button.grid(row=2, column=1, padx=10, pady=10, sticky="w")

push_data_button = tk.Button(setup_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=8, pady=6,
                             text="Push Data", width=12, command=push_data)
push_data_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")

edit_workflow_button = tk.Button(setup_frame, font=FONT_BTN, relief='flat', borderwidth=0,
                                 padx=8, pady=6, text="Edit Workflow", width=12,
                                 command=open_in_editor)
edit_workflow_button.grid(row=3, column=1, padx=10, pady=10, sticky="w")

# %%
# ACTION BUTTONS - below the notebook, so they are visible from every tab

load_button = tk.Button(action_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=10, pady=6, text="Load params", command=load_csv, width=14)
load_button.grid(row=0, column=0, padx=10, pady=5)

overwrite_button = tk.Button(action_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=10, pady=6, text="Overwrite params", command=overwrite_csv, width=14)
overwrite_button.grid(row=1, column=0, padx=10, pady=5)

run_protocol_button = tk.Button(action_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=10, pady=6, text="Launch Bonsai", state='active',
                                command=launch_bonsai, width=12)
run_protocol_button.grid(row=0, column=1, padx=10, pady=5)

kill_bonsai_button = tk.Button(action_frame, font=FONT_BTN, relief='flat', borderwidth=0, padx=10, pady=6, text="Kill Bonsai", state='active',
                               command=kill_bonsai, width=12)
kill_bonsai_button.grid(row=1, column=1, padx=10, pady=5)

rig_label = tk.Label(action_frame, text='', font=FONT_RIG, **({'bg':CANVAS} if CANVAS else {}))
rig_label.grid(row=2, column=0, columnspan=2, pady=(8, 0))

status_label = tk.Label(action_frame, text='', font=('Segoe UI', 11), fg=COL_MUTE, **({'bg':CANVAS} if CANVAS else {}))
status_label.grid(row=3, column=0, columnspan=2, pady=(2, 0))

_paint(flush_rig_button, COL_OK)
_paint(camera_button,    COL_OK)
_paint(push_data_button, COL_OK)
_paint(kill_bonsai_button, COL_STOP)
_paint(edit_workflow_button, COL_ACCENT)

# Resolve which rig this machine is BEFORE anything can launch. On failure, disable
# launch entirely and put the reason in the status line - the rig must be known.
try:
    RIG = resolve_rig()
    rig_label.config(text="Rig %s   (%s)" % (RIG['rig'], RIG['Hostname']), fg="#2e7d32")
except Exception as e:
    RIG = None
    run_protocol_button.config(state='disabled')
    _paint(run_protocol_button, COL_IDLE)
    tk.messagebox.showerror("Rig not configured", str(e))
    rig_label.config(text="RIG NOT CONFIGURED - launch disabled", fg="#c62828")

refresh_buttons()
root.mainloop()
