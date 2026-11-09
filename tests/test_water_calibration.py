"""Checks for GUI/water_calibration.py (the Calibrate button) and the two water-calibration workflows.

Runs off-rig, without Bonsai or a display:
    python -m pytest tests
"""
import csv
import importlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'GUI'))
wc = importlib.import_module('water_calibration')

NS = '{https://bonsai-rx.org/2018/workflow}'
XSI = '{http://www.w3.org/2001/XMLSchema-instance}type'


def test_volume_and_suggestion():
    """50 openings of 5 µl weigh 0.25 g; 0.27 g from 200 openings at 30 ms is 1.35 µl, so 5 µl needs ~111 ms."""
    assert wc.COUNT == 50 and wc.ul_per_opening(0.25) == pytest.approx(5.0)
    assert wc.ul_per_opening(0.27, 200) == pytest.approx(1.35)
    assert wc.suggested_ms(30, 1.35) == 111
    assert wc.in_range(4.5) and wc.in_range(5.5) and not wc.in_range(4.4) and not wc.in_range(5.6)


def test_a_third_run_is_offered_when_two_runs_differ():
    assert not wc.runs_disagree([1.00, 1.05])
    assert wc.runs_disagree([1.00, 1.20])
    assert not wc.runs_disagree([1.00])


def test_runs_alternate_between_the_spouts():
    assert wc.run_plan() == [('left', 1), ('right', 1), ('left', 2), ('right', 2)]


def test_new_valve_times():
    """A spout in range keeps its time; out of range it moves to its suggestion; two spouts further apart
    than 0.5 µl both move."""
    ok_left = wc.Spout('left', 100, [0.25, 0.25])           # 5.0 µl from 50 openings
    low_right = wc.Spout('right', 100, [0.20, 0.20])        # 4.0 µl
    assert wc.new_valve_times(ok_left, low_right) == {'Left_Valve_Time': 100, 'Right_Valve_Time': 125}
    assert wc.needs_change(ok_left, low_right)
    left = wc.Spout('left', 100, [0.23, 0.23])              # 4.6 µl
    right = wc.Spout('right', 100, [0.27, 0.27])            # 5.4 µl: both in range, 0.8 µl apart
    assert wc.new_valve_times(left, right) == {'Left_Valve_Time': 109, 'Right_Valve_Time': 93}
    same = wc.Spout('right', 100, [0.255, 0.255])          # 5.1 µl
    assert not wc.needs_change(ok_left, same)
    assert wc.new_valve_times(ok_left, same) == {'Left_Valve_Time': 100, 'Right_Valve_Time': 100}


def test_bonsai_command():
    harp = wc.bonsai_command('Bonsai.exe', 'w.bonsai', 1, 75, harp_port='COM6')
    assert harp[:3] == ['Bonsai.exe', 'w.bonsai', '--start'] and '--no-editor' not in harp
    assert {'-p:ValveNumber=1', '-p:ValveTime=75', '-p:Count=50', '-p:PortName=COM6'} <= set(harp)
    arduino = wc.bonsai_command('Bonsai.exe', 'w.bonsai', 2, 110.5)
    assert '-p:ValveTime=110.5' in arduino and not any(a.startswith('-p:PortName') for a in arduino)


def test_each_rig_gets_its_workflow():
    for row, name in (({'Arduino': 'TRUE'}, 'Manual_Calibration_Arduino.bonsai'),
                      ({'Arduino': 'FALSE'}, 'Manual_Calibration_Harp.bonsai')):
        path = wc.workflow_for(row, REPO)
        assert path.name == name and path.exists()


def _externalised(path):
    nodes = list(ET.parse(path).getroot().find(NS + 'Workflow').find(NS + 'Nodes'))
    return {(p.get('DisplayName') or p.get('Name')) for n in nodes if n.get(XSI) == 'ExternalizedMapping'
            for p in n.findall(NS + 'Property')}, nodes


@pytest.mark.parametrize('name', ['Manual_Calibration_Harp.bonsai', 'Manual_Calibration_Arduino.bonsai'])
def test_workflows_take_the_settings_the_panel_passes(name):
    """The panel sets ValveNumber, ValveTime and Count with -p:Name=value, so the workflows must expose
    them under exactly these names; opened by hand they start at 30 ms and 50 openings."""
    path = REPO / 'Protocols' / 'Water_calibration' / name
    names, _nodes = _externalised(path)
    assert {'ValveNumber', 'ValveTime', 'Count'} <= names
    text = path.read_text(encoding='utf-8-sig')
    i = text.find('DisplayName="ValveTime"')
    assert '<Value>30</Value>' in text[i:i + 300]
    assert '<rx:Count>50</rx:Count>' in text and '<rx:Count>200</rx:Count>' not in text


def test_harp_workflow_takes_its_port_from_the_panel():
    """The Harp board's port is passed as -p:PortName: it must be externalised onto the Device node."""
    path = REPO / 'Protocols' / 'Water_calibration' / 'Manual_Calibration_Harp.bonsai'
    names, nodes = _externalised(path)
    assert 'PortName' in names
    wf = ET.parse(path).getroot().find(NS + 'Workflow')
    device = [i for i, n in enumerate(nodes) if n.find(NS + 'Combinator') is not None
              and n.find(NS + 'Combinator').get(XSI) == 'harp:Device']
    port = [i for i, n in enumerate(nodes) if n.get(XSI) == 'ExternalizedMapping'
            and any(p.get('Name') == 'PortName' for p in n.findall(NS + 'Property'))]
    edges = {(int(e.get('From')), int(e.get('To'))) for e in wf.find(NS + 'Edges')}
    assert len(device) == 1 and len(port) == 1 and (port[0], device[0]) in edges


def test_log_keeps_every_run(tmp_path):
    log = tmp_path / 'calibration' / 'water_calibration.csv'
    spout = wc.Spout('left', 30)
    wc.log_run(log, '31101', spout, 0.27)
    wc.log_run(log, '31101', spout, 0.28)
    with open(log) as f:
        rows = list(csv.DictReader(f))
    assert [r['grams'] for r in rows] == ['0.27', '0.28'] and rows[0]['ul_per_opening'] == '5.400'
    assert list(rows[0]) == wc.LOG_COLUMNS
