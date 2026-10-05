"""Consistency checks between the GUI, the parameter files and Sound_Cat_V2.bonsai.

Runs off-rig (no Bonsai, no hardware). Needs pandas, numpy, openpyxl and pytest:
    python -m pytest tests
"""
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'GUI'))
import params_io  # noqa: E402

WORKFLOW = REPO / 'Protocols' / 'Auditory_discrimination' / 'Sound_Cat_V2.bonsai'
XLSX = REPO / 'Params' / 'Mouse_Room_Params.xlsx'
RIGS = REPO / 'Params' / 'Rigs.csv'

B = '{https://bonsai-rx.org/2018/workflow}'
XSI_TYPE = '{http://www.w3.org/2001/XMLSchema-instance}type'

# Trial_Summary columns whose name differs from the subject they record.
COLUMN_RENAMES = {'Go_Cue_Duration': 'Go_Cue_Dur', 'Opto_On': 'Opto_ON'}
NEW_TRIAL_SUMMARY_COLUMNS = ['Session_Type', 'Stimulation_Site', 'Stimulation_Type']


# ---------------------------------------------------------------- workflow reading
def _local(tag):
    return tag.split('}', 1)[-1]


def _text(el, name):
    for child in el:
        if _local(child.tag) == name:
            return child.text
    return None


class Graph:
    """One nested <Workflow>: its nodes and its labelled edges."""

    def __init__(self, workflow_el):
        nodes_el = workflow_el.find(B + 'Nodes')
        self.nodes = list(nodes_el) if nodes_el is not None else []
        edges_el = workflow_el.find(B + 'Edges')
        self.edges = [(int(e.get('From')), int(e.get('To')), e.get('Label'))
                      for e in (edges_el if edges_el is not None else [])]

    def kind(self, i):
        el = self.nodes[i]
        kind = el.get(XSI_TYPE) or ''
        if kind == 'Combinator':
            return 'Combinator:' + (el.find(B + 'Combinator').get(XSI_TYPE) or '')
        return kind

    def name(self, i):
        return _text(self.nodes[i], 'Name')

    def inputs(self, i):
        """{label: source node} for node i."""
        return {label: src for src, dst, label in self.edges if dst == i}

    def outputs(self, i):
        return [dst for src, dst, _ in self.edges if src == i]

    def find(self, kind, name=None):
        return [i for i in range(len(self.nodes))
                if self.kind(i) == kind and (name is None or self.name(i) == name)]


def load_workflow():
    root = ET.parse(WORKFLOW).getroot()
    graphs = {}

    def walk(workflow_el, path):
        g = Graph(workflow_el)
        graphs[path] = g
        for i, node in enumerate(g.nodes):
            sub = node.find(B + 'Workflow')
            if sub is not None:
                walk(sub, path + ((i, _text(node, 'Name')),))

    walk(root.find(B + 'Workflow'), ())
    return graphs


def top_group(graphs, name):
    matches = [g for path, g in graphs.items() if len(path) == 1 and path[0][1] == name]
    assert len(matches) == 1, f"expected one top-level group called {name}"
    return matches[0]


def parsers(graphs):
    """Every 'param = ...' PythonTransform in Params: (key, type, source file, Equal operands)."""
    g = top_group(graphs, 'Params')
    found = []
    for i in g.find('ipy:PythonTransform'):
        script = _text(g.nodes[i], 'Script') or ''
        m = re.search(r"param = '([^']*)'", script)
        if not m:
            continue
        ptype = re.search(r"param_type = (\w+)", script).group(1)
        source = g.name(g.inputs(i)['Source1'])
        operands = []
        for j in g.outputs(i):
            if g.kind(j) == 'Equal':
                operand = [c for c in g.nodes[j] if _local(c.tag) == 'Operand'][0]
                operands.append(_text(operand, 'Value'))
        found.append((m.group(1), ptype, source, operands))
    return found


def bonsai_parse(line, key, ptype):
    """The IronPython parser template used by every param node, run in CPython."""
    start = line.find(key + ": ") + len(key + ": ")
    end = line.find(",", start)
    return {'float': float, 'str': str, 'int': int}[ptype](line[start:end])


# ---------------------------------------------------------------- GUI output
def subject_lines():
    df = pd.read_excel(XLSX, sheet_name='Params')
    for _, row in df.iterrows():
        yield row['Subject'], params_io.format_line(params_io.subject_pairs_from_row(row))


def rig_lines():
    with open(RIGS) as f:
        for row in csv.DictReader(f):
            yield row['rig'], params_io.format_line(params_io.rig_pairs_from_row(row))


@pytest.fixture(scope='module')
def graphs():
    return load_workflow()


# ---------------------------------------------------------------- tests
def test_every_parser_finds_its_key_and_branch(graphs):
    lines = {'Subject_Params': list(subject_lines()), 'Rig_Params': list(rig_lines())}
    failures = []
    for key, ptype, source, operands in parsers(graphs):
        for who, line in lines[source]:
            if line.find(key + ": ") < 0:
                failures.append(f"{source} for {who} has no '{key}'")
                continue
            try:
                value = bonsai_parse(line, key, ptype)
            except ValueError as e:
                failures.append(f"{key} for {who}: {e}")
                continue
            if operands and str(value) not in operands:
                failures.append(f"{key} for {who} = '{value}' matches neither {operands}")
    assert not failures, "\n".join(failures)


def test_every_gui_key_has_a_parser(graphs):
    parsed = {key for key, _, _, _ in parsers(graphs)}
    unparsed = [key for key, *_ in params_io.SPEC if key not in parsed]
    assert not unparsed, f"GUI writes these but the workflow never reads them: {unparsed}"


def _resolve(g, node, items):
    """Follow Item1/Item2... accessors from node back to the node that supplies the value."""
    while True:
        kind = g.kind(node)
        if kind == 'MemberSelector' and _text(g.nodes[node], 'Selector') == 'Value':
            node = g.inputs(node)['Source1']                  # unwraps a Timestamp
        elif kind in ('Combinator:rx:Timestamp', 'Combinator:rx:Sample'):
            node = g.inputs(node)['Source1']
        elif items and kind in ('Combinator:rx:CombineLatest', 'Combinator:rx:Zip'):
            label = 'Source' + items[0][len('Item'):]
            assert label in g.inputs(node), f"{kind} at node {node} has no {label}"
            node, items = g.inputs(node)[label], items[1:]
        else:
            return node, items


def test_trial_summary_columns_point_at_the_right_subjects(graphs):
    g = top_group(graphs, 'Wide_Form_Saving')
    writer = g.find('io:CsvWriter')[0]
    flatten = g.inputs(writer)['Source1']
    expression = _text(g.nodes[flatten], 'Expression')
    columns = re.findall(r"(Item\d(?:\.Item\d)*)\s+as\s+(\w+)", expression)
    zip_node = g.inputs(flatten)['Source1']
    failures = []
    for path, column in columns:
        node, rest = _resolve(g, zip_node, path.split('.'))
        if column == 'Trial_End_Time':
            continue
        if rest or g.kind(node) != 'SubscribeSubject':
            failures.append(f"{column}: {path} does not reach a SubscribeSubject")
            continue
        subject = g.name(node)
        if subject != COLUMN_RENAMES.get(column, column):
            failures.append(f"{column}: {path} reads subject '{subject}'")
    missing = [c for c in NEW_TRIAL_SUMMARY_COLUMNS if c not in {c for _, c in columns}]
    assert not failures, "\n".join(failures)
    assert not missing, f"Trial_Summary is missing columns: {missing}"


def test_every_csv_writer_closes_at_session_end(graphs):
    declared = [g.name(i) for g in graphs.values() for i in range(len(g.nodes))
                if g.kind(i) == 'rx:PublishSubject' and g.name(i) == 'Close_Files']
    assert declared, "no PublishSubject called Close_Files is declared"
    failures = []
    for path, g in graphs.items():
        for writer in g.find('io:CsvWriter'):
            where = '/'.join(name or str(i) for i, name in path) + f" (CsvWriter node {writer})"
            chain = [writer]
            while g.kind(chain[-1]) != 'Combinator:rx:TakeUntil' and len(g.outputs(chain[-1])) == 1:
                chain.append(g.outputs(chain[-1])[0])
            last = chain[-1]
            if g.kind(last) != 'Combinator:rx:TakeUntil':
                branches = len(g.outputs(last))
                failures.append(f"{where}: " + ("the chain branches before any TakeUntil" if branches > 1
                                                else "no TakeUntil after the writer"))
                continue
            if g.inputs(last).get('Source1') != chain[-2]:
                failures.append(f"{where}: the writer chain must be Source1 of the TakeUntil")
            stop = g.inputs(last).get('Source2')
            if stop is None or g.kind(stop) != 'SubscribeSubject' or g.name(stop) != 'Close_Files':
                failures.append(f"{where}: Source2 of the TakeUntil is not SubscribeSubject Close_Files")
    assert not failures, "\n".join(failures)
