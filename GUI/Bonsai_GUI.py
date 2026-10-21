"""Control panel for the room-311 rigs.

Loads a mouse's parameters, starts and stops the Bonsai task workflow, and runs the utilities on
the Setup tab. Started on a rig by Launch_Bonsai_GUI.bat.

Code that needs no window lives in params_io.py (parameters, presets, rig identity) and
push_data.py (copying sessions to the server), so the tests can run it off-rig.
"""
import platform
import socket
import subprocess
import time
import tkinter as tk
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import psutil
from pythonosc.udp_client import SimpleUDPClient

import params_io
import push_data

# ================================================================ paths and constants
GUI_DIR = Path(__file__).resolve().parent
REPO = GUI_DIR.parent
WORKFLOW = REPO / 'Protocols' / 'Auditory_discrimination' / 'Sound_Cat_V2.bonsai'
CAMERA_WORKFLOW = REPO / 'Params' / 'Camera.bonsai'
FLUSH_WORKFLOW = REPO / 'Params' / 'Flush_Rig.bonsai'             # rigs whose valves run on the Arduino
FLUSH_WORKFLOW_HARP = REPO / 'Params' / 'Flush_Rig_Harp.bonsai'  # rigs with the Harp behaviour board
MOUSE_PARAMS_XLSX = REPO / 'Params' / 'Mouse_Room_Params.xlsx'
PRESETS_CSV = REPO / 'Params' / 'Presets.csv'
RIGS_CSV = REPO / 'Params' / 'Rigs.csv'
SUBJECT_PARAMS_CSV = REPO / 'Params' / 'Subject_Params.csv'   # written here, read by the workflow
RIG_PARAMS_CSV = REPO / 'Params' / 'Rig_Params.csv'           # written here, read by the workflow
EXPERIMENTERS_DIR = GUI_DIR / 'paths'
DATA_DIR = REPO / 'Data'
RIG_JSON = Path(r'C:\ProgramData\MouseRoom\rig.json')         # {"rig": "<id>"}, one per rig PC
BONSAI_EXE = Path.home() / 'AppData' / 'Local' / 'Bonsai' / 'Bonsai.exe'

# Bonsai resolves the workflow's relative paths (..\Params\..., Extensions\..., ../Data) against
# its working directory, so every Bonsai started from here runs in GUI/.
BONSAI_CWD = GUI_DIR
OSC_ADDRESS = ('127.0.0.1', 1334)   # the workflow listens here for /GUI End_Protocol
CLOSE_FILES_WAIT_S = 4              # the workflow closes its CSVs 2 s after End_Protocol
SELECT = 'Select'                   # shown in a field nobody has set

# ================================================================ appearance
IS_MAC = platform.system() == 'Darwin'
COL_OK = '#4c956c'          # buttons: go / loaded / ready
COL_WARN = '#e8a44c'        # buttons: needs attention
COL_STOP = '#c44536'        # buttons: running / stop
COL_IDLE = '#9aa0a6'        # buttons: not ready
COL_ACCENT = '#2f3e46'      # labels and field text
COL_MUTE = '#8a9199'        # secondary text
COL_CHANGED_TEXT = '#b35c00'
COL_RIG_OK = '#2e7d32'
COL_RIG_BAD = '#c62828'
FIELD_BLANK = '#f4f6f7'     # fields: not set yet
FIELD_ROW = '#f2cc55'       # fields: loaded from the mouse's own row (yellow)
FIELD_PRESET = '#a9cce3'    # fields: loaded from the mouse's preset (blue)
FIELD_CHANGED = '#f4a582'   # fields: changed in the panel after loading (orange)
# The rigs run Windows in light mode: an off-white canvas with white cards. On macOS the
# backgrounds are left to the system theme so they follow dark mode.
CANVAS = None if IS_MAC else '#eef1f4'
CARD = None if IS_MAC else '#ffffff'
FONT_UI = ('Segoe UI', 12)
FONT_BTN = ('Segoe UI', 12, 'bold')
FONT_TAB = ('Segoe UI', 11)
FONT_SMALL = ('Segoe UI', 11)


def bg(colour):
    """Background option for a frame or label; empty on macOS so the system theme applies."""
    return {'bg': colour} if colour else {}


def shade(hex_colour, factor=0.88):
    """A darker #rrggbb, for the pressed state."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % (int(r * factor), int(g * factor), int(b * factor))


def paint_button(button, colour):
    """Button colours show state. macOS (Aqua) ignores a button's bg; highlightbackground tints it."""
    if IS_MAC:
        button.config(highlightbackground=colour, activebackground=colour, fg='#ffffff')
    else:
        button.config(bg=colour, activebackground=shade(colour), fg='#ffffff',
                      disabledforeground='#eef1f4')


def paint_field(dropdown, colour):
    """Field colours show where a value came from; the text stays dark so it stays readable."""
    if IS_MAC:
        dropdown.config(highlightbackground=colour)
    else:
        dropdown.config(bg=colour, activebackground=shade(colour),
                        fg=COL_ACCENT, activeforeground=COL_ACCENT)


def warn(message):
    messagebox.showwarning('Warning', message)


# ================================================================ state
@dataclass
class State:
    saved: dict = None           # the values last written to Subject_Params.csv (Load or Overwrite)
    mouse: object = None         # params_io.Mouse from the last Load, to show and confirm hand changes
    running: bool = False        # a session is running (End Session not pressed yet)
    ended_at: float = None       # time.time() when End_Protocol went out; None once Bonsai is closed
    closing: bool = False        # Kill Bonsai is waiting for the data files to close
    camera_on: bool = False
    flush_on: bool = False


S = State()
process = None   # the Bonsai this panel started last (session, Camera or Flush Rig)
RIG = None       # this PC's Rigs.csv row, set at start-up; None if the PC is not a registered rig


# ================================================================ parameters
def current_values():
    """{xlsx column: text} as the fields show them now."""
    return {col: WIDGETS[key].var.get() for key, col, _, _, _ in SPEC}


def hand_changes(values):
    """Fields changed in the panel since Load: [(label, loaded text, current text, where it came from)]."""
    if S.mouse is None:
        return []
    changes = []
    for _, col, label, _, _ in SPEC:
        before, now = S.mouse.values[col], values[col]
        if now != before:
            source = f'preset {S.mouse.preset}' if col in S.mouse.from_preset else 'own row'
            changes.append((label, before, now, source))
    return changes


def preset_label_warning(values):
    """A paragraph for the Overwrite question when Session Type still names the loaded preset but a
    value that preset sets has been changed, so the data would carry a preset name that no longer
    describes the session. Empty otherwise."""
    m = S.mouse
    if m is None or m.preset == params_io.CUSTOM or values['Session_Type'] != m.preset:
        return ''
    differ = [label for _, col, label, _, _ in SPEC
              if col in m.from_preset and col != 'Session_Type' and values[col] != m.values[col]]
    if not differ:
        return ''
    return (f"Session Type still says '{m.preset}', but this session no longer matches that preset "
            f"({', '.join(differ)}). The data will be labelled '{m.preset}'; change Session Type "
            'if it should say something else.\n\n')


def repaint_fields():
    """Blue: from the preset. Yellow: from the mouse's row. Orange: changed here since Load."""
    values = current_values()
    for key, col, _, _, _ in SPEC:
        value = values[col]
        if value == SELECT:
            colour = FIELD_BLANK
        elif S.mouse is None or value != S.mouse.values[col]:
            colour = FIELD_CHANGED
        elif col in S.mouse.from_preset:
            colour = FIELD_PRESET
        else:
            colour = FIELD_ROW
        paint_field(WIDGETS[key].dropdown, colour)


def on_field_changed():
    repaint_fields()
    refresh()


def load_params():
    subject = WIDGETS['Animal_ID'].var.get()
    if subject == SELECT:
        warn('Select a subject')
        return
    try:
        mouse = params_io.load_mouse(MOUSE_PARAMS_XLSX, PRESETS_CSV, subject)
    except params_io.SpreadsheetError as e:
        warn(str(e))
        return
    if mouse.problems:
        warn('Not loaded - fix the spreadsheet:\n' + '\n'.join(mouse.problems))
        return

    params_io.write_subject_params(SUBJECT_PARAMS_CSV, mouse.pairs())
    for key, col, _, _, _ in SPEC:
        WIDGETS[key].var.set(mouse.values[col])    # exactly the text just written
    S.mouse, S.saved = mouse, dict(mouse.values)
    repaint_fields()
    refresh()
    if mouse.preset == params_io.CUSTOM:
        source = 'All values are from its row in the spreadsheet (custom).'
    else:
        source = (f"{len(mouse.from_preset)} values are from preset '{mouse.preset}' (blue), "
                  'the rest from its row in the spreadsheet (yellow).')
        if mouse.overridden:
            hidden = [f'  {col}: {own}   (preset: {used})' for col, own, used in mouse.overridden[:10]]
            if len(mouse.overridden) > 10:
                hidden.append(f'  ... and {len(mouse.overridden) - 10} more')
            source += (f"\n\nNot used from {mouse.subject}'s row, because preset '{mouse.preset}' "
                       'sets them:\n' + '\n'.join(hidden))
    messagebox.showinfo('Success', f'Params successfully loaded\n\n{source}')


def overwrite_params():
    values = current_values()
    if SELECT in values.values():
        warn('All params must be filled in')
        return
    if S.mouse is not None and values['Subject'] != S.mouse.subject:
        warn(f"Subject was changed after loading {S.mouse.subject}.\n"
             f"Press Load params to load {values['Subject']}'s settings.")
        return
    problems = params_io.check_values(values)
    if problems:
        warn('Not saved:\n' + '\n'.join(problems))
        return
    changes = hand_changes(values)
    if changes:
        listing = '\n'.join(f'{label}: {before} -> {now}   ({source})'
                            for label, before, now, source in changes)
        if not messagebox.askyesno('Save changes for this session?',
                                   f'These differ from what was loaded:\n\n{listing}\n\n'
                                   f'{preset_label_warning(values)}'
                                   'They apply to this session only; the spreadsheet is not changed.'):
            return

    params_io.write_subject_params(SUBJECT_PARAMS_CSV, params_io.pairs_from_values(values))
    S.saved = values
    refresh()
    messagebox.showinfo('Success', 'Overwrite successful')


# ================================================================ Bonsai
def bonsai_running():
    """True if any Bonsai is open on this PC, whoever started it."""
    return any((p.info['name'] or '').lower() == 'bonsai.exe'
               for p in psutil.process_iter(['name']))


def refuse_if_bonsai_open():
    """Refuse (with a warning) while anything Bonsai is open: by the panel's own state, and by
    process name in case Bonsai was started some other way."""
    if S.ended_at is not None or S.camera_on or S.flush_on or bonsai_running():
        warn('Bonsai is already open (a session, Camera, Flush Rig or the editor).\n'
             'Press Kill Bonsai first.')
        return True
    return False


def start_bonsai(workflow, start=True):
    """Start Bonsai on a workflow in BONSAI_CWD. Returns the process, or None after a warning."""
    if not workflow.exists():
        warn(f'{workflow.name} not found at {workflow}')
        return None
    if not BONSAI_EXE.exists():
        warn(f'Bonsai.exe not found at {BONSAI_EXE}')
        return None
    return subprocess.Popen([str(BONSAI_EXE), str(workflow)] + (['--start'] if start else []),
                            cwd=BONSAI_CWD)


def launch_or_end():
    """The Launch Bonsai button, which reads End Session while a session runs."""
    global process
    if S.running:
        end_session()
        return
    if RIG is None:
        warn('This PC is not registered as a rig, so it cannot run a session '
             '(see the red line at the bottom).')
        return
    if S.saved is None:
        warn("Protocol can't launch without params!")
        return
    if current_values() != S.saved:
        warn('Params changed since last save - press Overwrite first')
        return
    if refuse_if_bonsai_open():
        return
    started = start_bonsai(WORKFLOW)
    if started is None:
        return
    process = started
    S.running, S.ended_at = True, None
    refresh()


def end_session():
    SimpleUDPClient(*OSC_ADDRESS).send_message('/GUI', 'End_Protocol')
    S.running, S.ended_at = False, time.time()
    refresh()


def kill_bonsai():
    """Stop Bonsai without losing the end of the data files.

    Killing Bonsai discards whatever its CSV writers still hold in memory, so if a session is
    running, End_Protocol goes out first and the kill waits until the workflow has closed its
    files."""
    if S.running:
        end_session()
    wait_s = 0 if S.ended_at is None else CLOSE_FILES_WAIT_S - (time.time() - S.ended_at)
    if wait_s > 0:
        S.closing = True
        refresh()
        root.after(int(wait_s * 1000), terminate_bonsai)
    else:
        terminate_bonsai()


def terminate_bonsai():
    """Close every Bonsai on this PC: the session, Camera, Flush Rig and any editor."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    for p in psutil.process_iter(['name']):
        if (p.info['name'] or '').lower() == 'bonsai.exe':
            try:
                p.kill()
            except psutil.Error:
                pass
    S.ended_at, S.closing, S.camera_on, S.flush_on = None, False, False, False
    refresh()


def toggle_camera():
    global process
    if S.camera_on:
        terminate_bonsai()
        return
    if refuse_if_bonsai_open():
        return
    started = start_bonsai(CAMERA_WORKFLOW)
    if started is not None:
        process, S.camera_on = started, True
        refresh()


def toggle_flush():
    global process
    if S.flush_on:
        terminate_bonsai()
        return
    if RIG is None:
        warn('This PC is not registered as a rig, so Flush Rig does not know its ports.')
        return
    if refuse_if_bonsai_open():
        return
    workflow = FLUSH_WORKFLOW if params_io.uses_arduino(RIG) else FLUSH_WORKFLOW_HARP
    started = start_bonsai(workflow)
    if started is not None:
        process, S.flush_on = started, True
        refresh()


def open_in_editor():
    """Open the task workflow in the Bonsai editor, in the working directory a session uses.
    (Started from the Start menu, Bonsai runs in AppData\\Local\\Bonsai, where neither
    ..\\Params\\ nor Extensions\\ resolves.)"""
    start_bonsai(WORKFLOW, start=False)


# ================================================================ Push Data
def push():
    initials = experimenter.var.get()
    config = EXPERIMENTERS.get(initials)
    if config is None:
        warn('Select an experimenter before pushing')
        return
    if S.running or bonsai_running():
        warn('End the session and press Kill Bonsai before pushing, '
             'so that no data file is still being written.')
        return
    if not DATA_DIR.is_dir():
        messagebox.showinfo('Push Data', 'There is no Data folder on this PC yet.')
        return
    destination = push_data.find_server_folder(config['data_path'])
    if destination is None:
        warn('No server found on current machine')
        return

    status_label.config(text='Copying data to the server...')
    root.update_idletasks()
    try:
        report = push_data.copy_new_files(DATA_DIR, destination, prefix=initials)
    except OSError as e:
        messagebox.showerror('Push Data stopped',
                             f'{e}\n\nEvery file copied before this point is complete. '
                             'Press Push Data again to carry on.')
    else:
        show = messagebox.showwarning if report.different else messagebox.showinfo
        show('Push Data', report.summary(destination))
    finally:
        refresh()


# ================================================================ display
def preset_line(values):
    """The line under the rig: the loaded mouse's preset, and what was changed by hand."""
    if S.mouse is None:
        return {'text': '', 'fg': COL_MUTE}
    m = S.mouse
    if m.preset == params_io.CUSTOM:
        text = f'{m.subject}: custom - every value from its spreadsheet row'
    else:
        text = f'{m.subject}: preset {m.preset} - {len(m.from_preset)} values from Presets.csv (blue)'
    changed = [label for label, _, _, _ in hand_changes(values)]
    if not changed:
        return {'text': text, 'fg': COL_ACCENT}
    shown = ', '.join(changed[:4]) + (f' and {len(changed) - 4} more' if len(changed) > 4 else '')
    return {'text': f'{text}\nChanged for this session: {shown}', 'fg': COL_CHANGED_TEXT}


def refresh():
    """The one place where state becomes button colours and status text."""
    values = current_values()
    saved = S.saved is not None
    dirty = saved and values != S.saved
    if S.running:
        launch_button.config(text='End Session', state='normal')
        paint_button(launch_button, COL_STOP)
        load_button.config(state='disabled')
        overwrite_button.config(state='disabled')
    else:
        launch_button.config(text='Launch Bonsai', state='normal' if RIG is not None else 'disabled')
        paint_button(launch_button, COL_OK if RIG is not None and saved and not dirty else COL_IDLE)
        load_button.config(state='normal')
        overwrite_button.config(state='normal')
    paint_button(load_button, COL_OK if saved else COL_WARN)
    paint_button(overwrite_button, COL_WARN if dirty or not saved else COL_OK)
    paint_button(camera_button, COL_STOP if S.camera_on else COL_OK)
    paint_button(flush_button, COL_STOP if S.flush_on else COL_OK)
    status_label.config(text='Closing data files...' if S.closing else
                        'Bonsai running' if S.running else
                        'Session ended - press Kill Bonsai' if S.ended_at is not None else
                        'Unsaved changes - press Overwrite' if dirty else
                        'Ready to launch' if saved else
                        'Load or Overwrite params to begin')
    preset_label.config(**preset_line(values))


# ================================================================ window
Field = namedtuple('Field', 'var label dropdown')


def make_field(parent, label, options, row):
    var = tk.StringVar(value=SELECT)
    text = tk.Label(parent, text=label + ':', width=24, font=FONT_UI, fg=COL_ACCENT,
                    anchor='e', justify='right', **bg(CARD))
    text.grid(row=row, column=0, padx=(10, 14), pady=5, sticky='e')
    dropdown = tk.OptionMenu(parent, var, *(options or [SELECT]),
                             command=lambda _choice: on_field_changed())
    dropdown.config(height=1, width=16, font=FONT_UI, relief='flat',
                    highlightthickness=1, highlightbackground='#c8ccd0', borderwidth=0)
    paint_field(dropdown, FIELD_BLANK)
    dropdown.grid(row=row, column=1, padx=10, pady=5, sticky='w')
    return Field(var, text, dropdown)


def make_button(parent, text, command=None, width=12, padx=8):
    return tk.Button(parent, text=text, command=command, width=width, font=FONT_BTN,
                     relief='flat', borderwidth=0, padx=padx, pady=6)


root = tk.Tk()
root.title('Bonsai Launcher GUI')
root.config(**bg(CANVAS))
root.geometry('880x740')
root.minsize(860, 640)

# The action bar packs first, at the bottom, so it keeps its strip when the notebook expands.
action_frame = tk.Frame(root, **bg(CANVAS))
action_frame.pack(side='bottom', pady=10)
notebook = ttk.Notebook(root)
notebook.pack(side='top', pady=10, padx=10, fill='both', expand=True)

style = ttk.Style()
try:
    style.theme_use('clam')
except tk.TclError:
    pass
style.configure('TNotebook', background=CANVAS or '#eef1f4', borderwidth=0, tabmargins=[10, 8, 10, 0])
style.configure('TNotebook.Tab', font=FONT_TAB, padding=[16, 8], borderwidth=0,
                background='#dfe3e8', foreground=COL_MUTE)
style.map('TNotebook.Tab', background=[('selected', CARD or '#ffffff')],
          foreground=[('selected', COL_ACCENT)], expand=[('selected', [0, 0, 0, 0])])

# The subject list comes from the spreadsheet. If it cannot be read, the panel still opens,
# so Kill Bonsai and Push Data keep working.
try:
    SUBJECT_LIST = params_io.subjects(params_io.read_mice(MOUSE_PARAMS_XLSX))
except params_io.SpreadsheetError as e:
    SUBJECT_LIST = []
    messagebox.showerror('Spreadsheet not readable', str(e))
SPEC = params_io.build_spec(SUBJECT_LIST)
EXPERIMENTERS = push_data.load_experimenters(EXPERIMENTERS_DIR)

frames = {}
for tab in params_io.TABS:
    page = tk.Frame(notebook, **bg(CARD))
    notebook.add(page, text=tab)
    frames[tab] = tk.Frame(page, **bg(CARD))
    frames[tab].pack(pady=(24, 0), anchor='n')

WIDGETS = {}
for tab in params_io.TABS:
    for row, (key, _, label, options, _) in enumerate(s for s in SPEC if s[4] == tab):
        WIDGETS[key] = make_field(frames[tab], label, params_io.option_texts(key, options), row)

# ---- Setup tab
setup = frames['Setup']
experimenter = make_field(setup, 'Experimenter', sorted(EXPERIMENTERS), 0)
flush_button = make_button(setup, 'Flush Rig', toggle_flush)
flush_button.grid(row=1, column=0, padx=10, pady=10, sticky='w')
camera_button = make_button(setup, 'Camera', toggle_camera)
camera_button.grid(row=2, column=0, padx=10, pady=10, sticky='w')
test_speakers_button = make_button(setup, 'Test Speakers', width=15)    # not implemented
test_speakers_button.config(state='disabled')
test_speakers_button.grid(row=1, column=1, padx=10, pady=10, sticky='w')
calibrate_button = make_button(setup, 'Calibrate', width=15)            # not implemented
calibrate_button.config(state='disabled')
calibrate_button.grid(row=2, column=1, padx=10, pady=10, sticky='w')
push_button = make_button(setup, 'Push Data', push)
push_button.grid(row=3, column=0, padx=10, pady=10, sticky='w')
edit_button = make_button(setup, 'Edit Workflow', open_in_editor)
edit_button.grid(row=3, column=1, padx=10, pady=10, sticky='w')

# ---- action bar, visible from every tab
load_button = make_button(action_frame, 'Load params', load_params, width=14, padx=10)
load_button.grid(row=0, column=0, padx=10, pady=5)
overwrite_button = make_button(action_frame, 'Overwrite params', overwrite_params, width=14, padx=10)
overwrite_button.grid(row=1, column=0, padx=10, pady=5)
launch_button = make_button(action_frame, 'Launch Bonsai', launch_or_end, padx=10)
launch_button.grid(row=0, column=1, padx=10, pady=5)
kill_button = make_button(action_frame, 'Kill Bonsai', kill_bonsai, padx=10)
kill_button.grid(row=1, column=1, padx=10, pady=5)
rig_label = tk.Label(action_frame, text='', font=FONT_BTN, **bg(CANVAS))
rig_label.grid(row=2, column=0, columnspan=2, pady=(8, 0))
preset_label = tk.Label(action_frame, text='', font=FONT_SMALL, wraplength=820, justify='center', **bg(CANVAS))
preset_label.grid(row=3, column=0, columnspan=2, pady=(2, 0))
status_label = tk.Label(action_frame, text='', font=FONT_SMALL, fg=COL_MUTE, **bg(CANVAS))
status_label.grid(row=4, column=0, columnspan=2, pady=(2, 0))

paint_button(push_button, COL_OK)
paint_button(kill_button, COL_STOP)
paint_button(edit_button, COL_ACCENT)

# ================================================================ start-up
# Find out which rig this PC is before anything can launch. If that fails, Launch stays disabled
# and Rig_Params.csv is deleted, so neither a session nor Flush Rig can run on a stale file
# (on a cloned disk it would hold the other rig's ports and valve times).
try:
    RIG, rig_pairs = params_io.resolve_rig(RIG_JSON, RIGS_CSV, socket.gethostname())
    params_io.write_rig_params(RIG_PARAMS_CSV, rig_pairs)
    rig_label.config(text=f"Rig {RIG['rig']}   ({RIG['Hostname']})", fg=COL_RIG_OK)
except (params_io.RigError, OSError) as e:
    RIG = None
    try:
        RIG_PARAMS_CSV.unlink(missing_ok=True)
    except OSError:
        pass
    rig_label.config(text='RIG NOT CONFIGURED - launch disabled', fg=COL_RIG_BAD)
    messagebox.showerror('Rig not configured', str(e))

refresh()
root.mainloop()
