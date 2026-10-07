"""Parameter table and the two CSV lines the Bonsai workflow reads.

Kept free of tkinter so the same code can be imported by the GUI and by
tests/test_rig_consistency.py. Bonsai parses each value with
    value.find(key + ": ") ... value.find(",", start)
so a missing key or a comma inside a value silently yields a wrong slice;
check_pairs() catches both before anything is written.
"""
import csv

import numpy as np

# Option lists reused across several params.
EPOCHS = ['Sound', 'Delay', 'Air_Puff', 'Go_Cue', 'Response_Window',
          'Feedback', 'Reward', 'Timeout', 'Inter_Trial_Interval']
PROPS = ['NaN', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
BOOL = ['True', 'False']

TABS = ['Setup', 'Session', 'Stimulus', 'Timing', 'Contingency',
        'Anti-bias', 'Opto', 'Opto Timing', 'Debug']

# Placeholder for the subject list, which comes from the spreadsheet at run time.
SUBJECTS = None

# SPEC is the single source of truth. Row order sets the key order in
# Subject_Params.csv; the 'tab' column sets where the widget appears.
SPEC = [
    # key                             xlsx column                       label                                options                                  cast   tab
    ('Animal_ID',                     'Subject',                        "Subject:",                          SUBJECTS,                                None,  'Session'),
    ('Protocol',                      'Protocol',                       "Protocol:",                         ["SOUND_CAT_DISC", "SOUND_CAT_CONT", "PRO_ANTI", "SOUND_CAT"], None, 'Session'),
    ('Stage',                         'Stage',                          "Stage:",                            ['Habituation', 'Lick_To_Release', 'Three_And_Three', 'Full_Task_Disc', 'Full_Task_Cont', 'Habituation_cont', 'Lick_To_Release_cont'], int, 'Session'),
    ('Session_Type',                  'Session_Type',                   "Session Type:",                     ['regular', 'opto', 'masking', 'washout', 'alm_control_uni', 'alm_control_bi'], str, 'Session'),
    ('Distribution',                  'Distribution',                   "Distribution:",                     ['NaN', 'Uniform', 'Asym_Left', 'Asym_Right'], None, 'Stimulus'),
    ('Sound_Duration',                'Sound_Duration',                 "Sound Duration:",                   [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], None, 'Stimulus'),
    ('Nb_Of_Stim',                    'Nb_Of_Stim',                     "Nb Of Stim:",                       [np.nan, 2, 4, 6, 8],                    int,   'Stimulus'),
    ('Stim_Type',                     'Stim_Type',                      "Stim Type:",                        ['NaN', 'PT', 'WN'],                     None,  'Stimulus'),
    ('AntiBias',                      'AntiBias',                       "AntiBias:",                         BOOL,                                    str,   'Anti-bias'),
    ('Emulator',                      'Emulator',                       "Emulator:",                         BOOL,                                    str,   'Debug'),
    ('Air_Puff_Contingency_Rule',     'Air_Puff_Contingency_Rule',      "Rule:",                             ['NaN', 'Pro_Only', 'Anti_Only', 'Blocks_30', 'Blocks_15', 'Random_Alternation'], None, 'Contingency'),
    ('Show_Contingency_Switches',     'Show_Contingency_Switches',      "Show Contingency \n Switches:",     BOOL,                                    str,   'Contingency'),
    ('Working_Memory_Type',           'Working_Memory_Type',            "Working Memory \n Type:",           ['NaN', 'Fixed', 'Variable'],            None,  'Contingency'),
    ('Sound_Air_Puff_Contingency',    'Sound_Air_Puff_Contingency',     "Sound Air \n Puff Contingency:",    ['Low_Pro_High_Anti', 'Low_Anti_High_Pro'], None, 'Contingency'),
    ('Sound_Contingency',             'Sound_Contingency',              "Sound \n Contingency:",             ['Low_Left_High_Right', 'Low_Right_High_Left'], None, 'Contingency'),
    ('Opto_ON',                       'Opto_ON',                        "Opto ON:",                          BOOL,                                    str,   'Opto'),
    ('Perc_Opto_Trials',              'Perc_Opto_Trials',               "% Trials:",                         np.arange(0, 110, 5),                    None,  'Opto'),
    ('Light_Freq (Hz)',               'Light_Freq (Hz)',                "Light Freq (Hz):",                  np.arange(0, 110, 10),                   None,  'Opto'),
    ('Opto_Onset_1',                  'Opto_Onset_1',                   "Onset_1:",                          EPOCHS,                                  None,  'Opto Timing'),
    ('Opto_Onset_2',                  'Opto_Onset_2',                   "Onset_2:",                          EPOCHS,                                  None,  'Opto Timing'),
    ('Opto_Offset_1',                 'Opto_Offset_1',                  "Offset_1:",                         EPOCHS,                                  None,  'Opto Timing'),
    ('Opto_Offset_2',                 'Opto_Offset_2',                  "Offset_2:",                         EPOCHS,                                  None,  'Opto Timing'),
    ('Opto_Duration',                 'Opto_Duration',                  "Duration:",                         np.arange(0, 1010, 100),                 None,  'Opto Timing'),
    ('Stimulation_Site',              'Stimulation_Site',               "Stim Site:",                        ['NaN', 'PPC', 'ACC', 'ALM'],            None,  'Opto'),
    ('Stimulation_Type',              'Stimulation_Type',               "Stim Type:",                        ['NaN', 'Unilateral_Left', 'Unilateral_Right', 'Bilateral'], None, 'Opto'),
    ('AntiBias_Exp_Rate',             'AntiBias_Exp_Rate',              "AB_Exp_Rate:",                      [np.nan, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  None,  'Anti-bias'),
    ('AntiBias_Window',               'AntiBias_Window',                "AB_Window:",                        [np.nan, 10, 20, 30, 40, 50],            int,   'Anti-bias'),
    ('AntiBias_Sigmoid_Slope',        'AntiBias_Sigmoid_Slope',         "AB_Slope:",                         [np.nan, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  None,  'Anti-bias'),
    ('Agent_Sim',                     'Agent_Sim',                      "Agent Sim:",                        BOOL,                                    str,   'Debug'),
    ('Agent_Performance',             'Agent_Performance',              "Agent \n Performance:",             PROPS,                                   None,  'Debug'),
    ('Agent_Bias',                    'Agent_Bias',                     "Agent Bias:",                       PROPS,                                   None,  'Debug'),
    ('Stim_Dur_Staircase',            'Stim_Dur_Staircase',             "Stim Dur Staircase:",               BOOL,                                    str,   'Stimulus'),
    ('Stim_Dur_Staircase_Perf_Thresh','Stim_Dur_Staircase_Perf_Thresh', "Stim Dur Staircase \n Perf Thresh:", PROPS,                                  None,  'Stimulus'),
    ('Stim_Dur_Staircase_Step',       'Stim_Dur_Staircase_Step',        "Stim Dur Staircase \n Step:",       ['NaN', '10', '20', '30', '40', '50'],   None,  'Stimulus'),
    ('Min_Stim_Dur',                  'Min_Stim_Dur',                   "Min Stim Dur:",                     ['NaN', '50', '100', '150', '200', '250', '300'], None, 'Stimulus'),
    ('Opto_Type',                     'Opto_Type',                      "Opto Type:",                        ['NaN', 'Zapit', 'Fiber'],               None,  'Opto'),
    ('Zapit_Nb_Conditions',           'Zapit_Nb_Conditions',            "Zapit Nb \n Conditions:",           ['NaN', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  int,   'Opto'),
    ('Inter_Trial_Interval',          'Inter_Trial_Interval',           "Inter Trial \n Interval:",          np.arange(0, 11, 1),                     None,  'Timing'),
    ('Timeout_Duration',              'Timeout_Duration',               "Timeout \n Duration:",              np.arange(0, 11, 1),                     None,  'Timing'),
    ('Response_Window',               'Response_Window',                "Response \n Window:",               np.arange(0, 11, 1),                     None,  'Timing'),
    ('Stim_Range_Min',                'Stim_Range_Min',                 "Stim Range \n Min:",                np.arange(40, 100, 1),                   int,   'Stimulus'),
    ('Stim_Range_Max',                'Stim_Range_Max',                 "Stim Range \n Max:",                np.arange(40, 1000, 1),                  int,   'Stimulus'),
    ('Go_Cue_Duration',               'Go_Cue_Duration',                "Delay before \n window (ms):",       np.arange(0, 1050, 50),                  None,  'Timing'),
    ('Early_Lick_Abort',              'Early_Lick_Abort',               "Early Lick \n Abort:",               BOOL,                                    str,   'Timing'),
    ('Go_Cue_Sound',                  'Go_Cue_Sound',                   "Go Cue \n Tone:",                    BOOL,                                    str,   'Timing'),
    ('Go_Cue_Freq',                   'Go_Cue_Freq',                    "Go Cue \n Freq (Hz):",               np.arange(1000, 20500, 500),             None,  'Timing'),
    ('Go_Cue_Level',                  'Go_Cue_Level',                   "Go Cue \n Level (dB):",              np.arange(40, 91, 1),                    None,  'Timing'),
    ('Go_Cue_Sound_Duration',         'Go_Cue_Sound_Duration',          "Go Cue Tone \n Duration (ms):",      np.arange(20, 1010, 10),                 None,  'Timing'),
    ('Visualiser_Window_Size',        'Visualiser_Window_Size',         "Visualiser \n Window Size:",        np.arange(10, 50, 5),                    int,   'Debug'),
    ('Stable_Start',                  'Stable_Start',                   "Stable Start:",                     BOOL,                                    str,   'Anti-bias'),
    ('Stable_Start_Window',           'Stable_Start_Window',            "Stable Start \n Window:",           np.arange(10, 55, 5),                    int,   'Anti-bias'),
    ('Max_Trials_Consec',             'Max_Trials_Consec',              "Max Trials \n Consec:",             np.arange(2, 11, 1),                     int,   'Anti-bias'),
    ('Stable_Stim_Dist_Boundary',     'Stable_Stim_Dist_Boundary',      "Stable Stim \n Dist Boundary:",     np.arange(0, 1, 0.1),                    None,  'Anti-bias'),
]

# Keys the workflow turns into a boolean by comparing the text with 'True' and
# 'False' (case-sensitive). Any other value matches neither branch, the subject
# never emits, and everything waiting on it (including Params) waits forever.
BOOL_KEYS = [key for key, _, _, options, _, _ in SPEC if options is BOOL]

# Columns written to Rig_Params.csv, in order. Every one is written even when
# blank ("Harp_Beh_Port: ,") so the Bonsai parser always finds its key.
RIG_PARAM_COLS = ['Room_ID', 'Rig_ID', 'Arduino', 'Harp_Beh_Port', 'Sound_Card_Port',
                  'Left_Valve_Time', 'Right_Valve_Time', 'Speaker_Slope',
                  'Speaker_Y_Intercept', 'Arduino_Port', 'Arduino_Mega_Port']
RIG_BOOL_KEYS = ['Arduino']


def build_spec(subject_options):
    """SPEC with the subject list filled in (the GUI calls this once at start-up)."""
    return [(key, col, label, subject_options if options is SUBJECTS else options, cast, tab)
            for key, col, label, options, cast, tab in SPEC]


def normalise_bool(value):
    """Excel and Rigs.csv write TRUE/FALSE; the workflow compares with True/False."""
    text = str(value).strip()
    if text.lower() == 'true':
        return 'True'
    if text.lower() == 'false':
        return 'False'
    return text


def format_line(pairs):
    """'Key: value, Key: value,' - the flat format the Bonsai parsers read."""
    return ", ".join(f"{key}: {value}" for key, value in pairs) + ","


def subject_pairs_from_row(row):
    """(key, value) pairs for Subject_Params.csv from one spreadsheet row (a dict-like)."""
    pairs = []
    for key, col, _, _, _, _ in SPEC:
        value = row[col]
        pairs.append((key, normalise_bool(value) if key in BOOL_KEYS else value))
    return pairs


def rig_pairs_from_row(row):
    """(key, value) pairs for Rig_Params.csv from one Rigs.csv row (a dict)."""
    return [(col, normalise_bool(row.get(col, '')) if col in RIG_BOOL_KEYS else row.get(col, ''))
            for col in RIG_PARAM_COLS]


def check_pairs(pairs, bool_keys):
    """List of problems that would make the Bonsai parsers read the wrong thing."""
    problems = []
    for key, value in pairs:
        text = str(value)
        if key in bool_keys and text not in BOOL:
            problems.append(f"{key} must be True or False (got '{text}')")
        if ',' in text:
            problems.append(f"{key} contains a comma, which the workflow cannot parse ('{text}')")
    return problems


def write_subject_params(path, pairs):
    """Same bytes as before: one csv.writer row holding the whole line (so it is quoted)."""
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([format_line(pairs)])


def write_rig_params(path, pairs):
    with open(path, 'w') as f:
        f.write(format_line(pairs))
