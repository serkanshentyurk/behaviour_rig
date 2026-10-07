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
LAYOUT = WORKFLOW.with_name(WORKFLOW.name + '.layout')          # what Bonsai 2.7 reads
NEW_STYLE_LAYOUT = WORKFLOW.parent / '.bonsai' / 'Settings' / (WORKFLOW.stem + '.layout')  # Bonsai 2.8+
XLSX = REPO / 'Params' / 'Mouse_Room_Params.xlsx'
RIGS = REPO / 'Params' / 'Rigs.csv'

B = '{https://bonsai-rx.org/2018/workflow}'
XSI_TYPE = '{http://www.w3.org/2001/XMLSchema-instance}type'

# Trial_Summary columns whose name differs from the subject they record.
COLUMN_RENAMES = {'Go_Cue_Duration': 'Go_Cue_Dur', 'Opto_On': 'Opto_ON'}
NEW_TRIAL_SUMMARY_COLUMNS = ['Session_Type', 'Stimulation_Site', 'Stimulation_Type',
                             'Window_Open_Time', 'First_Lick_Time', 'Early_Lick_Time']
TRIAL_TIMES = ['Window_Open_Time', 'First_Lick_Time', 'Early_Lick_Time']
FULL_TASK_STAGES = ['Full_Task_Disc', 'Full_Task_Cont']
FRESH_LICK_STAGES = ['Three_And_Three', 'Full_Task_Disc', 'Full_Task_Cont']


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
        if i is None or i < 0:
            return None
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


def group(graphs, *names):
    """Nested workflow reached through group names (stage Conditions share names, so skip them)."""
    path = ()
    for name in names:
        g = graphs[path]
        hits = [i for i in range(len(g.nodes))
                if g.name(i) == name and g.kind(i) != 'rx:Condition' and path + ((i, name),) in graphs]
        assert len(hits) == 1, f"expected one group called {name} under {path}"
        path += ((hits[0], name),)
    return graphs[path]


def upstream(g, node, steps):
    """Kinds of the nodes reached by following Source1 back from node."""
    kinds = []
    for _ in range(steps):
        node = g.inputs(node)['Source1']
        kinds.append((g.kind(node), node))
    return kinds


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


def test_bonsai_can_compile_the_csharp_operators():
    """The GUI starts Bonsai in GUI/. Bonsai only compiles GUI/Extensions/*.cs (zapit_TCPclient)
    when GUI/Extensions.csproj sits next to that folder; without it the Zapit node is an
    unknown type and the workflow will not start, emulator or not."""
    scripts = sorted((REPO / 'GUI' / 'Extensions').glob('*.cs'))
    assert scripts, "no C# operators found in GUI/Extensions"
    assert (REPO / 'GUI' / 'Extensions.csproj').exists(), \
        "GUI/Extensions.csproj is missing, so Bonsai will not compile " + ", ".join(f.name for f in scripts)


def test_visualiser_layout_lines_up_with_the_workflow(graphs):
    """The .layout file holds one entry per node, in node order. If it is out of step with the
    workflow, or has no visualisers, the trial/performance windows silently stop appearing."""
    failures, windows = [], []

    def check(layout_el, path):
        g = graphs[path]
        entries = [c for c in layout_el if c.tag == 'DialogSettings']
        if len(entries) != len(g.nodes):
            failures.append(f"{'/'.join(n or str(i) for i, n in path) or 'top level'}: "
                            f"{len(entries)} layout entries for {len(g.nodes)} nodes")
            return
        for i, entry in enumerate(entries):
            child = path + ((i, g.name(i)),)
            if entry.findtext('VisualizerTypeName'):
                windows.append(g.name(i) or g.kind(i))
                if child in graphs and not graphs[child].find('WorkflowOutput'):
                    failures.append(f"group {g.name(i)} has a visualiser but no WorkflowOutput, so it shows nothing")
            nested = entry.find('EditorVisualizerLayout')
            if nested is not None and child in graphs:
                check(nested, child)

    check(ET.parse(LAYOUT).getroot(), ())
    assert not failures, "\n".join(failures)
    assert windows, f"{LAYOUT.name} has no visualiser windows"
    if NEW_STYLE_LAYOUT.exists():
        assert ET.parse(NEW_STYLE_LAYOUT).getroot().find('DialogSettings') is not None, \
            f"{NEW_STYLE_LAYOUT} is empty; Bonsai 2.8+ would use it instead of {LAYOUT.name} and open no windows"


def test_responses_only_count_licks_that_start_while_listening(graphs):
    """The beams are BehaviorSubjects, so subscribing replays the current state; Skip(1) drops it,
    so a contact already in progress when the rig starts listening is not taken as the choice."""
    failures = []
    for stage in FRESH_LICK_STAGES:
        g = group(graphs, 'Trial', 'Sound_Cat_Trial', stage, 'Response')
        for beam in ('Beam_1_Bool', 'Beam_2_Bool'):
            for s in g.find('SubscribeSubject', beam):
                outs = g.outputs(s)
                skip = outs[0] if len(outs) == 1 else None
                if skip is None or g.kind(skip) != 'Combinator:rx:Skip' or \
                        _text(g.nodes[skip].find(B + 'Combinator'), 'Count') != '1':
                    failures.append(f"{stage}/Response: {beam} is not followed by Skip(1)")
    assert not failures, "\n".join(failures)


def _is_window_opening(g, i):
    """True if node i fires when Trial_Epoch becomes Response_Window (Condition <- Equal <- SubscribeSubject)."""
    if i is None or g.kind(i) != 'rx:Condition':
        return False
    eq = g.inputs(i).get('Source1')
    if eq is None or g.kind(eq) != 'Equal':
        return False
    if [_text(c, 'Value') for c in g.nodes[eq] if _local(c.tag) == 'Operand'] != ['Response_Window']:
        return False
    src = g.inputs(eq).get('Source1')
    return g.kind(src) == 'SubscribeSubject' and g.name(src) == 'Trial_Epoch'


def test_trial_times_are_declared_reset_and_set(graphs):
    failures = []
    variables = group(graphs, 'Variables')
    for t in TRIAL_TIMES:
        decl = variables.find('rx:BehaviorSubject', t)
        if len(decl) != 1 or _text(variables.nodes[variables.inputs(decl[0])['Source1']].find(B + 'Combinator'),
                                   'Value') != 'NaN':
            failures.append(f"Variables: {t} is not declared once with a NaN start value")
    for stage in FULL_TASK_STAGES:
        g = group(graphs, 'Trial', 'Sound_Cat_Trial', stage)
        # reset at the start of every trial
        for t in TRIAL_TIMES:
            writers = g.find('MulticastSubject', t)
            if not any(g.kind(g.inputs(w)['Source1']) == 'Combinator:DoubleProperty' and
                       _text(g.nodes[g.inputs(w)['Source1']].find(B + 'Combinator'), 'Value') == 'NaN' for w in writers):
                failures.append(f"{stage}: {t} is not reset to NaN at the start of the trial")
        # window open / first lick: timestamps of Record_Latency's two inputs
        rl = group(graphs, 'Trial', 'Sound_Cat_Trial', stage, 'Record_Latency')
        for t, source in (('Window_Open_Time', 'Source1'), ('First_Lick_Time', 'Source2')):
            ok = False
            for w in rl.find('MulticastSubject', t):
                chain = upstream(rl, w, 3)
                ok |= ([k for k, _ in chain[:2]] == ['MemberSelector', 'Combinator:rx:Timestamp'] and
                       chain[2][0] == 'WorkflowInput' and rl.name(chain[2][1]) == source)
            if not ok:
                failures.append(f"{stage}/Record_Latency: {t} is not the timestamp of {source}")
        # early lick: first new contact, cut off when the window opens (Go_Cue output)
        go_cue = g.find('rx:SelectMany', 'Go_Cue')
        ok = False
        for w in g.find('MulticastSubject', 'Early_Lick_Time'):
            if g.kind(g.inputs(w)['Source1']) != 'MemberSelector':
                continue
            chain = upstream(g, w, 4)
            if [k for k, _ in chain] == ['MemberSelector', 'Combinator:rx:Timestamp', 'Combinator:rx:Take',
                                         'Combinator:rx:TakeUntil']:
                ok |= _is_window_opening(g, g.inputs(chain[3][1]).get('Source2'))
        if not ok:
            failures.append(f"{stage}: Early_Lick_Time is not the first lick before the window opens")
    assert not failures, "\n".join(failures)


def test_response_latency_comes_from_the_clock_times(graphs):
    g = top_group(graphs, 'Wide_Form_Saving')
    writer = g.find('io:CsvWriter')[0]
    flatten = g.inputs(writer)['Source1']
    expression = _text(g.nodes[flatten], 'Expression')
    m = re.search(r"\(\s*(Item\d(?:\.Item\d)*)\s*-\s*(Item\d(?:\.Item\d)*)\s*\)\s*\*\s*1000\s+as\s+Response_Latency",
                  expression)
    assert m, "Response_Latency is not (First_Lick_Time - Window_Open_Time) * 1000"
    zip_node = g.inputs(flatten)['Source1']
    names = [g.name(_resolve(g, zip_node, path.split('.'))[0]) for path in m.groups()]
    assert names == ['First_Lick_Time', 'Window_Open_Time'], names


def test_each_spout_uses_its_own_valve_time(graphs):
    """In SOUND_CAT, a group that opens the right valve must time it with Right_Valve_Time, and the
    left with Left_Valve_Time. (Until 5 Oct 2026 every right reward used Left_Valve_Time.)"""
    trial = [path for path in graphs if [n for _, n in path][:2] == ['Trial', 'Sound_Cat_Trial']]
    failures, checked = [], 0
    for path in trial:
        g = graphs[path]
        opened = {g.name(i) for i in g.find('MulticastSubject')
                  if g.name(i) in ('Left_Valve', 'Right_Valve')
                  and g.kind(g.inputs(i).get('Source1', -1)) == 'Combinator:BooleanProperty'
                  and _text(g.nodes[g.inputs(i)['Source1']].find(B + 'Combinator'), 'Value') == 'true'}
        if not opened:
            continue
        checked += 1
        times = {g.name(i) for i in g.find('SubscribeSubject') if (g.name(i) or '').endswith('_Valve_Time')}
        if times != {v + '_Time' for v in opened}:
            failures.append(f"{'/'.join(n or '' for _, n in path)}: opens {sorted(opened)} but is timed by {sorted(times)}")
    assert checked >= 10, f"expected at least 10 SOUND_CAT reward groups, found {checked}"
    assert not failures, "\n".join(failures)


def test_stimulus_stage_ends_only_after_the_sound(graphs):
    """Full_Task_Cont/Stim/Normal_Stim must finish only when Play has played and waited out the sound.
    Until 5 Oct 2026 a branch from Correct_Side also ended it, so with Asym_Left/Asym_Right the go cue
    started at sound onset and some trials played no sound at all."""
    g = group(graphs, 'Trial', 'Sound_Cat_Trial', 'Full_Task_Cont', 'Stim', 'Normal_Stim')
    out = g.find('WorkflowOutput')[0]
    upstream, todo = set(), [out]
    while todo:
        n = todo.pop()
        for src in g.inputs(n).values():
            if src not in upstream:
                upstream.add(src); todo.append(src)
    names = {(g.kind(i), g.name(i)) for i in upstream}
    assert ('rx:SelectMany', 'Play') in names, "Normal_Stim's output does not come from Play"
    assert ('MulticastSubject', 'Correct_Side') not in names, \
        "Normal_Stim's output also depends on Correct_Side, so the stage can end before the sound has played"
    # A one-input Merge means 'flatten a stream of streams' in Bonsai; fed an ordinary stream it fails to build.
    lonely = [i for i in upstream if g.kind(i) == 'Combinator:rx:Merge' and len(g.inputs(i)) < 2]
    assert not lonely, f"Normal_Stim has a Merge with one input on its output path (node {lonely}); Bonsai will reject it"


def _upstream(g, node):
    seen, todo = set(), [node]
    while todo:
        n = todo.pop()
        for src in g.inputs(n).values():
            if src not in seen:
                seen.add(src); todo.append(src)
    return seen


def _path_of(graphs, g):
    return [path for path, gg in graphs.items() if gg is g][0]


def test_early_lick_abort_stops_the_trial(graphs):
    """With Early_Lick_Abort on, an early lick cuts the trial before Go_Cue and before the window,
    then runs Feedback -> the usual timeout -> outcome 'Early', and marks the trial as aborted."""
    failures = []
    for stage in FULL_TASK_STAGES:
        g = group(graphs, 'Trial', 'Sound_Cat_Trial', stage)
        stim, go = g.find('rx:SelectMany', 'Stim')[0], g.find('rx:SelectMany', 'Go_Cue')[0]
        flags = [i for i in g.find('rx:Condition')
                 if g.kind(g.inputs(i).get('Source1')) == 'MemberSelector'
                 and any(g.kind(u) == 'SubscribeSubject' and g.name(u) == 'Early_Lick_Abort' for u in _upstream(g, i))]
        if len(flags) != 1:
            failures.append(f"{stage}: no single early-abort signal gated by Early_Lick_Abort"); continue
        e = flags[0]
        cuts = [i for i in g.find('Combinator:rx:TakeUntil') if g.inputs(i).get('Source2') == e]
        before_go = [c for c in cuts if g.inputs(c).get('Source1') == stim and g.inputs(go).get('Source1') == c]
        after_go = [c for c in cuts if g.inputs(c).get('Source1') == go]
        if not before_go:
            failures.append(f"{stage}: Go_Cue is not cut off by the early-abort signal")
        targets = {g.name(t) or g.kind(t) for c in after_go for t in g.outputs(c)}
        if not {'Response', 'Record_Latency', 'Combinator:rx:Delay'} <= targets:
            failures.append(f"{stage}: the window is not cut off by the early-abort signal (cut feeds {sorted(targets)})")
        if any(src == go for t in g.find('rx:SelectMany', 'Response') for src in g.inputs(t).values()):
            failures.append(f"{stage}: Response is still fed by Go_Cue directly")
        early = [i for i in g.find('Combinator:StringProperty') if _text(g.nodes[i].find(B + 'Combinator'), 'Value') == 'Early']
        if len(early) != 1 or e not in _upstream(g, early[0]):
            failures.append(f"{stage}: no 'Early' outcome reached from the early-abort signal")
        else:
            up = _upstream(g, early[0])
            if not any(g.kind(u) == 'MulticastSubject' and g.name(u) == 'Trial_Epoch' for u in up):
                failures.append(f"{stage}: the early abort does not pass through the Feedback epoch")
            outcome_merge = [m for m in g.outputs(early[0]) if g.kind(m) == 'Combinator:rx:Merge']
            if not outcome_merge or not any(g.kind(t) == 'Combinator:rx:Take' for t in g.outputs(outcome_merge[0])):
                failures.append(f"{stage}: 'Early' does not reach the Trial_Outcome merge")
            timeout = g.inputs(early[0]).get('Source1')            # the group that feeds 'Early'
            sub = graphs.get(_path_of(graphs, g) + ((timeout, g.name(timeout)),)) if timeout is not None else None
            if sub is None or not any(sub.kind(i) == 'SubscribeSubject' and sub.name(i) == 'Timeout_Duration' for i in range(len(sub.nodes))):
                failures.append(f"{stage}: the early abort is not followed by the Timeout_Duration timeout")
        if not any(g.kind(t) == 'Combinator:BooleanProperty' and any(g.name(m) == 'Abort_Trial' for m in g.outputs(t))
                   for t in g.outputs(e)):
            failures.append(f"{stage}: the early abort does not set Abort_Trial")
    assert not failures, "\n".join(failures)

def test_no_group_contains_a_loop(graphs):
    """Bonsai can neither build nor draw a group whose connections form a loop; it fails at that group
    and the editor crashes when the group is opened. (Feedback has to go through a subject instead.)"""
    loops = []
    for path, g in graphs.items():
        n = len(g.nodes)
        nxt = {i: [] for i in range(n)}
        for a, b, _ in g.edges:
            nxt[a].append(b)
        state = [0] * n
        for start in range(n):
            if state[start]:
                continue
            stack = [(start, iter(nxt[start]))]
            state[start] = 1
            while stack:
                u, it = stack[-1]
                v = next(it, None)
                if v is None:
                    state[u] = 2; stack.pop()
                elif state[v] == 1:
                    loops.append('/'.join(name or str(i) for i, name in path) or 'top level')
                    break
                elif state[v] == 0:
                    state[v] = 1; stack.append((v, iter(nxt[v])))
            else:
                continue
            break
    assert not loops, f"connections form a loop in: {sorted(set(loops))}"
