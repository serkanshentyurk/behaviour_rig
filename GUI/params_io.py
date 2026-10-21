"""Parameters: the SPEC table, presets, rig identity, and the two one-line files the workflow reads.

Nothing here imports tkinter, so the panel (Bonsai_GUI.py) and the tests share this code.

The workflow parses every value with

    value.find(key + ": ") ... value.find(",", start)

so a missing key, or a comma in a value, silently yields a wrong slice. Everything is
checked before a file is written.

Where a mouse's values come from
--------------------------------
Every mouse has a full row in Params/Mouse_Room_Params.xlsx (sheet Params). Its Preset cell
says custom (or is empty), or names a preset:

  custom    the row is used as it is.
  a preset  for every parameter the preset sets, the preset's value from Params/Presets.csv is
            used and the row's is ignored (Load lists the row's values this hides); the other
            parameters still come from the row.
            A preset is named after a Session Type, and Session_Type is set to that name,
            so the data files record which preset ran.

Presets.csv has one row per preset: its name in the Preset column, an optional Description, and
a column for each parameter. An empty cell means the preset does not set that parameter; a cell
that says NaN sets it to empty.
"""
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- option lists
EMPTY = 'nan'      # how an empty value is written; the workflow reads it as NaN, or as text that matches nothing
BOOL = ['True', 'False']
EPOCHS = ['Sound', 'Delay', 'Air_Puff', 'Go_Cue', 'Response_Window',
          'Feedback', 'Reward', 'Timeout', 'Inter_Trial_Interval']
PROPS = [EMPTY, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

TABS = ['Setup', 'Session', 'Stimulus', 'Timing', 'Contingency',
        'Anti-bias', 'Opto', 'Opto Timing', 'Debug']

SUBJECTS = None    # placeholder: the subject list comes from the spreadsheet (see build_spec)


def steps(start, stop, step):
    """Evenly spaced options, stop excluded, without float noise (0.1 * 3 is 0.3, not 0.30000000000000004)."""
    return np.round(np.arange(start, stop, step), 6)


# The Session Types, which are also the preset names: each preset in Presets.csv is named after one.
# Saved in Session_Parameters and every Trial_Summary row; the workflow does not branch on it.
SESSION_TYPES = [
    'habituation', 'lick-to-release', 'three-and-three',
    'regular-antibias-200', 'regular-antibias-175', 'regular-antibias-150',
    'regular-antibias-100', 'regular-antibias-050', 'regular-antibias-025',
    'regular', 'hard-A', 'hard-B',
    'masking-regular', 'masking-hard-A', 'masking-hard-B', 'washout-regular',
    'opto-regular', 'opto-hard-A', 'opto-hard-B', 'control-alm-bi', 'control-alm-uni',
]


# SPEC is the single source of truth for the subject parameters. Row order sets the key order in
# Subject_Params.csv and the order of the fields on each tab. PARAMETERS.md documents every row;
# tests/test_docs.py fails if the two disagree.
SPEC = [
    # key                              xlsx column                        panel label                     options                                   tab
    ('Animal_ID',                      'Subject',                         'Subject',                      SUBJECTS,                                 'Session'),
    ('Protocol',                       'Protocol',                        'Protocol',                     ['SOUND_CAT', 'PRO_ANTI'],                'Session'),
    ('Stage',                          'Stage',                           'Stage',                        ['Habituation', 'Lick_To_Release', 'Three_And_Three', 'Full_Task_Disc', 'Full_Task_Cont'], 'Session'),
    ('Session_Type',                   'Session_Type',                    'Session Type',                 SESSION_TYPES,                            'Session'),
    ('Distribution',                   'Distribution',                    'Distribution',                 [EMPTY, 'Uniform', 'Asym_Left', 'Asym_Right'], 'Stimulus'),
    ('Sound_Duration',                 'Sound_Duration',                  'Sound Duration',               [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], 'Stimulus'),
    ('Nb_Of_Stim',                     'Nb_Of_Stim',                      'Nb Of Stim',                   [EMPTY, 2, 4, 6, 8],                      'Stimulus'),
    ('Stim_Type',                      'Stim_Type',                       'Stim Type',                    [EMPTY, 'PT', 'WN'],                      'Stimulus'),
    ('AntiBias',                       'AntiBias',                        'AntiBias',                     BOOL,                                     'Anti-bias'),
    ('Emulator',                       'Emulator',                        'Emulator',                     BOOL,                                     'Debug'),
    ('Air_Puff_Contingency_Rule',      'Air_Puff_Contingency_Rule',       'Rule',                         [EMPTY, 'Pro_Only', 'Anti_Only', 'Blocks_30', 'Blocks_15', 'Random_Alternation'], 'Contingency'),
    ('Show_Contingency_Switches',      'Show_Contingency_Switches',       'Show Contingency Switches',    BOOL,                                     'Contingency'),
    ('Working_Memory_Type',            'Working_Memory_Type',             'Working Memory Type',          [EMPTY, 'Fixed', 'Variable'],             'Contingency'),
    ('Sound_Air_Puff_Contingency',     'Sound_Air_Puff_Contingency',      'Sound Air Puff Contingency',   ['Low_Pro_High_Anti', 'Low_Anti_High_Pro'], 'Contingency'),
    ('Sound_Contingency',              'Sound_Contingency',               'Sound Contingency',            ['Low_Left_High_Right', 'Low_Right_High_Left'], 'Contingency'),
    ('Opto_ON',                        'Opto_ON',                         'Opto ON',                      BOOL,                                     'Opto'),
    ('Perc_Opto_Trials',               'Perc_Opto_Trials',                '% Trials',                     steps(0, 105, 5),                         'Opto'),
    ('Light_Freq (Hz)',                'Light_Freq (Hz)',                 'Light Freq (Hz)',              steps(0, 110, 10),                        'Opto'),
    ('Opto_Onset_1',                   'Opto_Onset_1',                    'Onset_1',                      EPOCHS,                                   'Opto Timing'),
    ('Opto_Onset_2',                   'Opto_Onset_2',                    'Onset_2',                      EPOCHS,                                   'Opto Timing'),
    ('Opto_Offset_1',                  'Opto_Offset_1',                   'Offset_1',                     EPOCHS,                                   'Opto Timing'),
    ('Opto_Offset_2',                  'Opto_Offset_2',                   'Offset_2',                     EPOCHS,                                   'Opto Timing'),
    ('Opto_Duration',                  'Opto_Duration',                   'Duration',                     steps(0, 1010, 100),                      'Opto Timing'),
    ('Stimulation_Site',               'Stimulation_Site',                'Stim Site',                    [EMPTY, 'PPC', 'ACC', 'ALM'],             'Opto'),
    ('Stimulation_Type',               'Stimulation_Type',                'Stim Type',                    [EMPTY, 'Unilateral_Left', 'Unilateral_Right', 'Bilateral'], 'Opto'),
    ('AntiBias_Exp_Rate',              'AntiBias_Exp_Rate',               'AB_Exp_Rate',                  [EMPTY, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],    'Anti-bias'),
    ('AntiBias_Window',                'AntiBias_Window',                 'AB_Window',                    [EMPTY, 10, 20, 30, 40, 50],              'Anti-bias'),
    ('AntiBias_Sigmoid_Slope',         'AntiBias_Sigmoid_Slope',          'AB_Slope',                     steps(0, 3.25, 0.25),                     'Anti-bias'),
    ('Agent_Sim',                      'Agent_Sim',                       'Agent Sim',                    BOOL,                                     'Debug'),
    ('Agent_Performance',              'Agent_Performance',               'Agent Performance',            PROPS,                                    'Debug'),
    ('Agent_Bias',                     'Agent_Bias',                      'Agent Bias',                   PROPS,                                    'Debug'),
    ('Stim_Dur_Staircase',             'Stim_Dur_Staircase',              'Stim Dur Staircase',           BOOL,                                     'Stimulus'),
    ('Stim_Dur_Staircase_Perf_Thresh', 'Stim_Dur_Staircase_Perf_Thresh',  'Stim Dur Staircase Perf Thresh', PROPS,                                  'Stimulus'),
    ('Stim_Dur_Staircase_Step',        'Stim_Dur_Staircase_Step',         'Stim Dur Staircase Step',      [EMPTY, 10, 20, 30, 40, 50],              'Stimulus'),
    ('Min_Stim_Dur',                   'Min_Stim_Dur',                    'Min Stim Dur',                 [EMPTY, 50, 100, 150, 200, 250, 300],     'Stimulus'),
    ('Opto_Type',                      'Opto_Type',                       'Opto Type',                    [EMPTY, 'Zapit', 'Fiber'],                'Opto'),
    ('Zapit_Nb_Conditions',            'Zapit_Nb_Conditions',             'Zapit Nb Conditions',          [EMPTY, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   'Opto'),
    ('Inter_Trial_Interval',           'Inter_Trial_Interval',            'Inter Trial Interval',         steps(0, 11, 1),                          'Timing'),
    ('Timeout_Duration',               'Timeout_Duration',                'Timeout Duration',             steps(0, 11, 1),                          'Timing'),
    ('Response_Window',                'Response_Window',                 'Response Window',              steps(0, 11, 1),                          'Timing'),
    ('Stim_Range_Min',                 'Stim_Range_Min',                  'Stim Range Min',               steps(40, 100, 1),                        'Stimulus'),
    ('Stim_Range_Max',                 'Stim_Range_Max',                  'Stim Range Max',               steps(40, 100, 1),                        'Stimulus'),
    ('Go_Cue_Duration',                'Go_Cue_Duration',                 'Delay before window (ms)',     steps(0, 1050, 50),                       'Timing'),
    ('Early_Lick_Abort',               'Early_Lick_Abort',                'Early Lick Abort',             BOOL,                                     'Timing'),
    ('Visualiser_Window_Size',         'Visualiser_Window_Size',          'Visualiser Window Size',       steps(10, 50, 5),                         'Debug'),
    ('Stable_Start',                   'Stable_Start',                    'Stable Start',                 BOOL,                                     'Anti-bias'),
    ('Stable_Start_Window',            'Stable_Start_Window',             'Stable Start Window',          steps(10, 55, 5),                         'Anti-bias'),
    ('Max_Trials_Consec',              'Max_Trials_Consec',               'Max Trials Consec',            steps(2, 11, 1),                          'Anti-bias'),
    ('Stable_Stim_Dist_Boundary',      'Stable_Stim_Dist_Boundary',       'Stable Stim Dist Boundary',    steps(0, 1, 0.1),                         'Anti-bias'),
]

SPEC_COLUMNS = [col for _, col, _, _, _ in SPEC]
KEY_OF = {col: key for key, col, _, _, _ in SPEC}

# Keys that may be empty (written 'nan'). The workflow ignores these when they do not apply,
# e.g. the opto settings when Opto ON is False. Every other key must have a value.
OPTIONAL_KEYS = {key for key, _, _, options, _ in SPEC
                 if isinstance(options, list) and EMPTY in options} | {
    'Perc_Opto_Trials', 'Light_Freq (Hz)', 'Opto_Duration', 'AntiBias_Sigmoid_Slope',
    'Opto_Onset_1', 'Opto_Onset_2', 'Opto_Offset_1', 'Opto_Offset_2'}

# Keys the workflow turns into a boolean by comparing the text with 'True' and 'False'
# (case-sensitive). Any other value matches neither branch, the subject never emits, and
# everything waiting on it (including Params) waits forever.
BOOL_KEYS = [key for key, _, _, options, _ in SPEC if options is BOOL]

# Columns written to Rig_Params.csv, in order. Every one is written even when blank
# ("Harp_Beh_Port: ,") so the Bonsai parser always finds its key.
RIG_PARAM_COLS = ['Room_ID', 'Rig_ID', 'Arduino', 'Harp_Beh_Port', 'Sound_Card_Port',
                  'Left_Valve_Time', 'Right_Valve_Time', 'Speaker_Slope',
                  'Speaker_Y_Intercept', 'Arduino_Port', 'Arduino_Mega_Port']
RIG_BOOL_KEYS = ['Arduino']

# Spreadsheet and preset file layout.
PARAMS_SHEET = 'Params'
PRESET_COL = 'Preset'            # in the Params sheet (if missing, every mouse is custom), and the
                                 # first column of Presets.csv
DESCRIPTION_COL = 'Description'  # optional column of Presets.csv, for people only
CUSTOM = 'custom'
NOT_IN_PRESETS = ['Subject', 'Session_Type']   # the mouse, and the preset's own name


def build_spec(subject_options):
    """SPEC with the subject list filled in (the panel calls this once at start-up)."""
    return [(key, col, label, subject_options if options is SUBJECTS else options, tab)
            for key, col, label, options, tab in SPEC]


# ---------------------------------------------------------------- one value as text
def as_text(value):
    """The text written for one cell.

    Whole numbers lose their '.0', other floats keep 10 significant digits, booleans are
    True/False, and an empty cell (or the text nan/NaN) is EMPTY. So the same value always
    gives the same text, whether Excel stored it as a number or as text.
    """
    if value is None:
        return EMPTY
    if isinstance(value, (bool, np.bool_)):
        return 'True' if value else 'False'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return EMPTY if math.isnan(value) else '%.10g' % value
    text = str(value).strip()
    return EMPTY if text == '' or text.lower() == 'nan' else text


def normalise_bool(text):
    """Excel and Rigs.csv write TRUE/FALSE; the workflow compares with True/False."""
    lower = str(text).strip().lower()
    return {'true': 'True', 'false': 'False'}.get(lower, str(text).strip())


def _is_number(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _kind(options):
    if options is SUBJECTS:
        return 'subject'
    if options is BOOL:
        return 'bool'
    real = [as_text(o) for o in options if as_text(o) != EMPTY]
    return 'number' if all(_is_number(o) for o in real) else 'choice'


KIND = {key: _kind(options) for key, _, _, options, _ in SPEC}
CHOICES = {key: [as_text(o) for o in options if as_text(o) != EMPTY]
           for key, _, _, options, _ in SPEC if KIND[key] == 'choice'}


def value_text(key, value):
    """as_text, plus the per-key clean-ups: True/False for yes/no keys, one spelling per number."""
    text = as_text(value)
    if KIND.get(key) == 'bool':
        return normalise_bool(text)
    if KIND.get(key) == 'number' and text != EMPTY and _is_number(text):
        return as_text(float(text))
    return text


def option_texts(key, options):
    """What the panel offers for a key: the options as text, with 'nan' first if the key may be empty."""
    texts = [] if options is None else [as_text(o) for o in options]
    if key in OPTIONAL_KEYS and EMPTY not in texts:
        texts.insert(0, EMPTY)
    return texts


def check_text(key, text):
    """A problem that would make the workflow's parser read the wrong slice of the line, or None."""
    if any(c in text for c in ',"\r\n'):
        return f"{key} contains a comma, quote or line break, which the workflow cannot parse ('{text}')"
    return None


def check_value(key, text):
    """None if the workflow can use this text for this key, else a short reason."""
    problem = check_text(key, text)
    if problem:
        return problem
    kind = KIND[key]
    if kind == 'bool':
        return None if text in BOOL else f"{key} must be True or False (got '{text}')"
    if text == EMPTY:
        return None if key in OPTIONAL_KEYS else f"{key} is empty"
    if kind == 'choice' and text not in CHOICES[key]:
        return f"{key} '{text}' is not one of: {', '.join(CHOICES[key])}"
    if kind == 'number' and not _is_number(text):
        return f"{key} '{text}' is not a number"
    return None


def check_values(values):
    """Problems with a full set of values ({xlsx column: text}), in SPEC order."""
    problems = []
    for key, col, _, _, _ in SPEC:
        problem = check_value(key, values.get(col, EMPTY))
        if problem:
            problems.append(problem)
    return problems


def pairs_from_values(values):
    """(key, value) pairs for Subject_Params.csv, in SPEC order."""
    return [(key, values[col]) for key, col, _, _, _ in SPEC]


# ---------------------------------------------------------------- mice and presets
class SpreadsheetError(Exception):
    """The spreadsheet or Presets.csv cannot be used at all; the message says why."""


@dataclass
class Preset:
    name: str
    values: dict                                   # xlsx column -> text, for the parameters it sets
    problems: list = field(default_factory=list)   # anything that stops a mouse loading with it


@dataclass
class Mouse:
    """The values one mouse runs with."""
    subject: str
    preset: str                                    # CUSTOM or the preset's name
    values: dict                                   # xlsx column -> text, every SPEC column
    from_preset: frozenset = frozenset()           # columns whose value came from the preset
    problems: list = field(default_factory=list)   # empty when the values can be written
    overridden: tuple = ()                         # (column, row's own value, value used) where the
                                                   # preset replaced a filled cell with something else

    def pairs(self):
        return pairs_from_values(self.values)


def _is_empty_cell(value):
    return as_text(value) == EMPTY


def read_mice(path):
    """The Params sheet, one row per mouse, each cell as Excel stored it."""
    try:
        # dtype=object: a whole number stays whole even in a column with gaps
        mice = pd.read_excel(path, sheet_name=PARAMS_SHEET, dtype=object)
    except (OSError, ValueError) as e:
        raise SpreadsheetError(f"Cannot read sheet '{PARAMS_SHEET}' of {path}:\n{e}")
    mice.columns = [str(c).strip() for c in mice.columns]
    if 'Subject' not in mice.columns:
        raise SpreadsheetError(f"Sheet '{PARAMS_SHEET}' of {path} has no Subject column")
    return mice[[not _is_empty_cell(s) for s in mice['Subject']]]


def subjects(mice):
    return [as_text(s) for s in mice['Subject']]


def read_presets(path):
    """{name: Preset} from Presets.csv (one row per preset); no file means no presets.

    A problem with the file as a whole (a misspelt column, a row without a name) is attached to
    every preset, so no mouse loads with a preset until the file is fixed. Custom mice are not
    affected."""
    if not Path(path).exists():
        return {}
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    except (OSError, UnicodeDecodeError) as e:
        raise SpreadsheetError(f"Cannot read {path}:\n{e}")
    if not rows:
        return {}

    header, body = rows[0], rows[1:]
    shared = []
    if header[0] != PRESET_COL:
        shared.append(f"the first column of {Path(path).name} must be called {PRESET_COL}")
    repeated = sorted({c for c in header if c and header.count(c) > 1})
    if repeated:
        shared.append(f"more than one column called {', '.join(repeated)}")
    columns = {}                                  # column index -> the parameter it sets
    for i, col in enumerate(header[1:], start=1):
        if col in (DESCRIPTION_COL, ''):
            continue
        if col in NOT_IN_PRESETS:
            shared.append(f"a preset cannot set {col}" +
                          (" (it is the preset's name)" if col == 'Session_Type' else ''))
        elif col not in SPEC_COLUMNS:
            shared.append(f"'{col}' is not a parameter column (misspelt?)")
        else:
            columns[i] = col
    names = [r[0] for r in body]
    if '' in names:
        shared.append(f"{names.count('')} row(s) have no name in the {PRESET_COL} column")

    presets = {}
    for r in body:
        name = r[0]
        if not name:
            continue
        own = []
        if names.count(name) > 1:
            own.append(f"the name '{name}' is used by more than one row")
        if name not in CHOICES['Session_Type']:
            own.append(f"'{name}' is not a Session Type; add it to SESSION_TYPES in params_io.py first")
        values = {}
        for i, col in columns.items():
            cell = r[i] if i < len(r) else ''
            if cell == '':
                continue
            key = KEY_OF[col]
            values[col] = value_text(key, cell)
            problem = check_value(key, values[col])
            if problem:
                own.append(problem)
        presets[name] = Preset(name, values, shared + own)
    return presets


def resolve(row, presets):
    """The values one mouse runs with: its row, with its preset's values replacing the row's."""
    subject = as_text(row.get('Subject'))
    name = as_text(row.get(PRESET_COL))
    problems, preset = [], None
    if name == EMPTY or name.lower() == CUSTOM:
        name = CUSTOM
    elif name not in presets:
        known = ', '.join(sorted(presets)) or 'none yet'
        problems.append(f"Preset '{name}' is not in Presets.csv (presets: {known})")
    else:
        preset = presets[name]
        problems += [f"Presets.csv, preset '{name}': {p}" for p in preset.problems]

    values, from_preset, overridden = {}, set(), []
    for key, col, _, _, _ in SPEC:
        own = value_text(key, row.get(col))
        if preset is not None and (col == 'Session_Type' or col in preset.values):
            used = name if col == 'Session_Type' else preset.values[col]
            values[col] = used
            from_preset.add(col)
            if own != EMPTY and own != used:
                overridden.append((col, own, used))
        else:
            values[col] = own
    problems += check_values(values)
    return Mouse(subject, name, values, frozenset(from_preset), problems, tuple(overridden))


def load_mouse(xlsx, presets_csv, subject):
    """One subject's values. Raises SpreadsheetError if the files cannot be used at all;
    otherwise check the returned Mouse's problems before writing anything."""
    mice = read_mice(xlsx)
    missing = [col for col in SPEC_COLUMNS if col not in mice.columns]
    if missing:
        raise SpreadsheetError("Spreadsheet is missing columns:\n" + "\n".join(missing))
    rows = mice[[s == subject for s in subjects(mice)]]
    if len(rows) == 0:
        raise SpreadsheetError("No params available for this subject")
    mouse = resolve(rows.iloc[0], read_presets(presets_csv))
    if len(rows) > 1:
        mouse.problems.insert(0, f"{subject} has {len(rows)} rows in the spreadsheet; keep one")
    return mouse


def resolve_all(xlsx, presets_csv):
    """Every mouse in the spreadsheet (used by the tests)."""
    presets = read_presets(presets_csv)
    return [resolve(row, presets) for _, row in read_mice(xlsx).iterrows()]


# ---------------------------------------------------------------- the rig
class RigError(Exception):
    """This PC cannot be matched to a row of Rigs.csv; the message says why."""


def read_rigs(path):
    with open(path, newline='') as f:
        # extra cells past the header land under the key None: ignore them, as before
        return [{k.strip(): (v or '').strip() for k, v in row.items() if k is not None}
                for row in csv.DictReader(f)]


def resolve_rig(rig_json, rigs_csv, hostname):
    """(row, pairs) for this PC. rig.json gives the rig id, Rigs.csv its row, and the row's
    Hostname must match this computer (catches a cloned disk that carried rig.json along)."""
    try:
        with open(rig_json) as f:
            rig_id = str(json.load(f)['rig']).strip()
    except FileNotFoundError:
        raise RigError(f"No rig.json on this machine (expected at {rig_json}).\n"
                       "This machine has not been registered as a rig.")
    except (ValueError, KeyError, TypeError):
        raise RigError(f'{rig_json} should contain {{"rig": "<id>"}}')

    rows = read_rigs(rigs_csv)
    matches = [r for r in rows if r.get('rig') == rig_id]
    if not matches:
        raise RigError(f"rig.json says '{rig_id}' but that is not a row in Rigs.csv.\n"
                       f"Known rigs: {', '.join(sorted(r.get('rig', '') for r in rows))}")
    if len(matches) > 1:
        raise RigError(f"Rigs.csv has {len(matches)} rows for rig '{rig_id}'; keep one.")
    row = matches[0]
    if row.get('Hostname') and row['Hostname'] != hostname:
        raise RigError(f"rig.json says '{rig_id}', whose Hostname is '{row['Hostname']}' in Rigs.csv,\n"
                       f"but this machine is '{hostname}'. Two machines sharing a rig id?")

    pairs = rig_pairs_from_row(row)
    problems = check_pairs(pairs, RIG_BOOL_KEYS)
    if problems:
        raise RigError(f"Rigs.csv row '{rig_id}' is not usable:\n" + "\n".join(problems))
    return row, pairs


def rig_pairs_from_row(row):
    """(key, value) pairs for Rig_Params.csv from one Rigs.csv row (a dict)."""
    return [(col, normalise_bool(row.get(col, '')) if col in RIG_BOOL_KEYS else row.get(col, ''))
            for col in RIG_PARAM_COLS]


def check_pairs(pairs, bool_keys):
    """Problems that would make the Bonsai parsers read the wrong thing."""
    problems = []
    for key, value in pairs:
        text = str(value)
        if key in bool_keys and text not in BOOL:
            problems.append(f"{key} must be True or False (got '{text}')")
        problem = check_text(key, text)
        if problem:
            problems.append(problem)
    return problems


# ---------------------------------------------------------------- the files Bonsai reads
def format_line(pairs):
    """'Key: value, Key: value,' - the flat format the Bonsai parsers read."""
    return ", ".join(f"{key}: {value}" for key, value in pairs) + ","


def write_subject_params(path, pairs):
    """One csv.writer row holding the whole line (so it is quoted), as the workflow expects."""
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([format_line(pairs)])


def write_rig_params(path, pairs):
    with open(path, 'w') as f:
        f.write(format_line(pairs))
