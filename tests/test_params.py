"""Checks for GUI/params_io.py and GUI/push_data.py, and for the files they read.

Runs off-rig, without Bonsai or a display:
    python -m pytest tests
"""
import csv
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'GUI'))
import params_io  # noqa: E402
import push_data  # noqa: E402

XLSX = REPO / 'Params' / 'Mouse_Room_Params.xlsx'
PRESETS = REPO / 'Params' / 'Presets.csv'
RIGS = REPO / 'Params' / 'Rigs.csv'
EXPERIMENTERS = REPO / 'GUI' / 'paths'


# ---------------------------------------------------------------- the repository's own files
def test_every_mouse_in_the_spreadsheet_loads():
    mice = params_io.resolve_all(XLSX, PRESETS)
    assert mice, "no mice in the spreadsheet"
    failures = [f"{m.subject}: {p}" for m in mice for p in m.problems]
    assert not failures, "\n".join(failures)
    names = [m.subject for m in mice]
    assert len(names) == len(set(names)), f"listed twice: {sorted({n for n in names if names.count(n) > 1})}"


def test_presets_file_is_usable():
    problems = [f"{name}: {p}" for name, preset in params_io.read_presets(PRESETS).items()
                for p in preset.problems]
    assert not problems, "\n".join(problems)


def test_every_preset_works_with_every_mouse():
    """A preset nobody uses yet must still give every mouse a set of values Load accepts."""
    presets = params_io.read_presets(PRESETS)
    failures = []
    for _, row in params_io.read_mice(XLSX).iterrows():
        for name in presets:
            trial = row.copy()
            trial[params_io.PRESET_COL] = name
            failures += [f"{row['Subject']} on {name}: {p}" for p in params_io.resolve(trial, presets).problems]
    assert not failures, "\n".join(failures)


def test_rigs_csv_rows_are_usable():
    rows = params_io.read_rigs(RIGS)
    for column in ('rig', 'Hostname'):
        values = [r[column] for r in rows]
        assert len(values) == len(set(values)), f"Rigs.csv repeats a {column}"
    failures = [f"{r['rig']}: {p}" for r in rows
                for p in params_io.check_pairs(params_io.rig_pairs_from_row(r), params_io.RIG_BOOL_KEYS)]
    assert not failures, "\n".join(failures)


def test_experimenter_files():
    experimenters = push_data.load_experimenters(EXPERIMENTERS)
    assert len(experimenters) == len(list(EXPERIMENTERS.glob('*.json'))), "a file in GUI/paths was skipped"
    for initials, config in experimenters.items():
        assert config['data_path'].startswith('/'), f"{initials}: data_path goes after the drive letter, so starts with /"


def test_no_key_is_the_end_of_another():
    """The workflow finds a value with line.find(key + ': '). If one key ended with another
    (Stim_Type and a new key Type, say), the shorter key would read the longer one's value."""
    for keys in ([k for k, *_ in params_io.SPEC], params_io.RIG_PARAM_COLS):
        clashes = [(a, b) for a in keys for b in keys if a != b and b.endswith(a)]
        assert not clashes, f"{clashes}: rename one, or the workflow may read the wrong value"


# ---------------------------------------------------------------- values
def test_values_have_one_spelling():
    t = params_io.value_text
    assert t('Nb_Of_Stim', 8.0) == t('Nb_Of_Stim', '8.0') == t('Nb_Of_Stim', 8) == '8'
    assert t('Stable_Stim_Dist_Boundary', 0.1 * 3) == '0.3'
    assert t('AntiBias', 'TRUE') == 'True' and t('AntiBias', False) == 'False'
    assert t('Opto_Type', float('nan')) == t('Opto_Type', 'NaN') == t('Opto_Type', None) == params_io.EMPTY


def test_options_in_the_panel_are_accepted_values():
    for key, _, _, options, _ in params_io.SPEC:
        for text in params_io.option_texts(key, options):
            assert params_io.check_value(key, text) is None, (key, text)
            if params_io.KIND[key] == 'number':
                assert len(text) < 12, (key, text)     # no float noise such as 0.30000000000000004


def test_bad_values_are_refused():
    v = params_io.check_value
    assert v('Stage', 'Full_task_Cont')                  # misspelt
    assert v('Inter_Trial_Interval', '2s')               # not a number
    assert v('Session_Type', params_io.EMPTY)            # must have a value
    assert v('Distribution', params_io.EMPTY) is None    # may be empty
    assert v('Opto_ON', params_io.EMPTY)                 # yes/no must be True or False
    assert v('Stimulation_Site', 'ALM, PPC')             # a comma breaks the workflow's parser


def test_load_then_overwrite_writes_the_same_file(tmp_path):
    """After Load the fields hold exactly the text Load wrote, so Overwrite without changes is
    accepted and writes the same bytes."""
    for mouse in params_io.resolve_all(XLSX, PRESETS):
        loaded, overwritten = tmp_path / 'load.csv', tmp_path / 'overwrite.csv'
        params_io.write_subject_params(loaded, mouse.pairs())
        shown = dict(mouse.values)
        assert not params_io.check_values(shown), mouse.subject
        params_io.write_subject_params(overwritten, params_io.pairs_from_values(shown))
        assert loaded.read_bytes() == overwritten.read_bytes(), mouse.subject


# ---------------------------------------------------------------- presets
def write_presets(path, rows):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    return params_io.read_presets(path)


def row_with(**cells):
    """A complete mouse row from the repository's spreadsheet, with some cells changed."""
    row = params_io.read_mice(XLSX).iloc[0].copy()
    for col, value in cells.items():
        row[col] = value
    return row


def test_custom_uses_the_row_and_a_preset_replaces_what_it_sets(tmp_path):
    presets = write_presets(tmp_path / 'Presets.csv', [
        ['Preset', 'Description', 'Opto_ON', 'Opto_Type', 'Zapit_Nb_Conditions'],
        ['opto-regular', 'light on, with a comma', 'TRUE', 'Zapit', '2'],
        ['regular', '', 'False', 'NaN', ''],
    ])
    assert not any(p.problems for p in presets.values())
    row = dict(Opto_ON='FALSE', Opto_Type='Fiber', Zapit_Nb_Conditions=5, Inter_Trial_Interval=7)

    custom = params_io.resolve(row_with(Preset='custom', **row), presets)
    assert custom.values['Opto_Type'] == 'Fiber' and not custom.from_preset and not custom.overridden

    opto = params_io.resolve(row_with(Preset='opto-regular', **row), presets)
    assert not opto.problems
    assert (opto.values['Opto_ON'], opto.values['Opto_Type'], opto.values['Zapit_Nb_Conditions']) == ('True', 'Zapit', '2')
    assert opto.values['Session_Type'] == 'opto-regular'  # the preset's name is the Session Type
    assert opto.values['Inter_Trial_Interval'] == '7'     # not set by the preset: from the row
    # the row's own values the preset hides, which Load lists
    assert {col for col, _, _ in opto.overridden} == {'Session_Type', 'Opto_ON', 'Opto_Type', 'Zapit_Nb_Conditions'}
    assert ('Opto_Type', 'Fiber', 'Zapit') in opto.overridden

    regular = params_io.resolve(row_with(Preset='regular', **row), presets)
    assert regular.values['Opto_Type'] == params_io.EMPTY      # NaN in a preset sets it to empty
    assert regular.values['Zapit_Nb_Conditions'] == '5'        # empty preset cell: from the row
    assert {col for col, _, _ in regular.overridden} == {'Session_Type', 'Opto_Type'}    # Opto_ON agrees
    empty_cell = params_io.resolve(row_with(Preset='opto-regular', Opto_Type=None), presets)
    assert 'Opto_Type' not in {col for col, _, _ in empty_cell.overridden}            # empty cells are not listed


def test_empty_preset_cell_is_custom_and_unknown_names_are_refused(tmp_path):
    presets = write_presets(tmp_path / 'Presets.csv', [['Preset', 'Opto_ON'], ['opto-regular', 'True']])
    assert params_io.resolve(row_with(Preset=None), presets).preset == params_io.CUSTOM
    misspelt = params_io.resolve(row_with(Preset='opto-regualr'), presets)
    assert misspelt.problems and 'opto-regular' in misspelt.problems[0]


def test_mistakes_in_the_presets_file_are_reported(tmp_path):
    presets = write_presets(tmp_path / 'Presets.csv', [
        ['Preset', 'Opto_on', 'Stage', 'Session_Type'],   # misspelt parameter; Session_Type is the name
        ['opto-regular', 'True', 'Full_task_Cont', ''],   # value not among the options
        ['my_preset', '', '', ''],                         # not a Session Type
        ['regular', '', '', ''],
        ['regular', '', '', ''],                           # the same name twice
    ])
    text = '\n'.join(presets['opto-regular'].problems)
    assert "'Opto_on' is not a parameter" in text and 'Full_task_Cont' in text and 'cannot set Session_Type' in text
    assert any('not a Session Type' in p for p in presets['my_preset'].problems)
    assert any('more than one row' in p for p in presets['regular'].problems)


# ---------------------------------------------------------------- rig identity
def test_rig_identity(tmp_path):
    rigs = tmp_path / 'Rigs.csv'
    rigs.write_text('rig,Hostname,Room_ID,Rig_ID,Arduino,Harp_Beh_Port,Sound_Card_Port,Arduino_Port,'
                    'Arduino_Mega_Port,Left_Valve_Time,Right_Valve_Time,Speaker_Slope,Speaker_Y_Intercept\n'
                    '1,PC-ONE,311,1,TRUE,,COM4,COM5,,100,90,1.3,-30\n')
    rig_json = tmp_path / 'rig.json'
    with pytest.raises(params_io.RigError, match='No rig.json'):
        params_io.resolve_rig(rig_json, rigs, 'PC-ONE')
    rig_json.write_text('{"rig": 1}')                                  # a number works as well as "1"
    _, pairs = params_io.resolve_rig(rig_json, rigs, 'PC-ONE')
    assert dict(pairs)['Arduino'] == 'True' and dict(pairs)['Harp_Beh_Port'] == ''
    with pytest.raises(params_io.RigError, match='Two machines'):    # a cloned disk
        params_io.resolve_rig(rig_json, rigs, 'PC-TWO')
    rig_json.write_text('{"rig": "2"}')
    with pytest.raises(params_io.RigError, match='not a row'):
        params_io.resolve_rig(rig_json, rigs, 'PC-ONE')


# ---------------------------------------------------------------- Push Data
def test_push_copies_only_what_is_missing(tmp_path):
    source, server = tmp_path / 'Data', tmp_path / 'server'
    for relative, text in {'SS1/s1/a.csv': 'aaa', 'SS1/s1/b.csv': 'bbb',
                           'SS1/s2/c.csv': 'ccc', 'QP1/s1/d.csv': 'ddd'}.items():
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_text(text)
    (server / 'SS1' / 's1').mkdir(parents=True)
    (server / 'SS1' / 's1' / 'a.csv').write_text('aaa')      # already there
    (server / 'SS1' / 's1' / 'b.csv').write_text('b')        # cut short by an earlier push

    report = push_data.copy_new_files(source, server, prefix='SS')
    assert report.copied == [os.path.join('SS1', 's2', 'c.csv')]
    assert report.already == 1 and report.different == [os.path.join('SS1', 's1', 'b.csv')]
    assert (server / 'SS1' / 's1' / 'b.csv').read_text() == 'b'           # never overwritten
    assert not (server / 'QP1').exists()                                   # another experimenter's mouse
    assert not list(server.rglob('*' + push_data.PARTIAL_SUFFIX))
