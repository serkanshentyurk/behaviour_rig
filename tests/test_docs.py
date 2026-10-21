"""PARAMETERS.md must list every parameter in GUI/params_io.py SPEC, with the same panel label,
spreadsheet column and options. The descriptions are written by hand; when SPEC changes, this
test says which table cell to update.
"""
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'GUI'))
import params_io  # noqa: E402

DOC = REPO / 'PARAMETERS.md'


def options_text(key, options):
    """How PARAMETERS.md writes a parameter's options."""
    if options is params_io.SUBJECTS:
        return 'subjects in the spreadsheet'
    if isinstance(options, np.ndarray):
        texts = [params_io.as_text(o) for o in options]
        text = f'{texts[0]}–{texts[-1]} in steps of {params_io.as_text(options[1] - options[0])}'
        return f'{params_io.EMPTY}, {text}' if key in params_io.OPTIONAL_KEYS else text
    return ', '.join(params_io.option_texts(key, options))


def doc_tables():
    """{tab: {panel label: (column, options)}} from the '## <tab> tab' sections."""
    tables, tab = {}, None
    for line in DOC.read_text(encoding='utf-8').splitlines():
        heading = re.match(r'## (.+) tab$', line)
        if heading:
            tab = heading.group(1)
            tables[tab] = {}
        elif line.startswith('## '):
            tab = None
        elif tab and line.startswith('|') and not line.startswith(('|---', '| Panel label')):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            tables[tab][cells[0]] = (cells[1].strip('`'), cells[2])
    return tables


def test_parameters_md_matches_spec():
    tables, problems = doc_tables(), []
    for key, col, label, options, tab in params_io.SPEC:
        row = tables.get(tab, {}).get(label)
        if row is None:
            problems.append(f"{tab} tab: no row for '{label}'")
            continue
        if row[0] != col:
            problems.append(f"{label}: the column should be `{col}` (it says {row[0]})")
        if row[1] != options_text(key, options):
            problems.append(f"{label}: Options should be '{options_text(key, options)}' (it says '{row[1]}')")
    in_spec = {(tab, label) for _, _, label, _, tab in params_io.SPEC}
    problems += [f"{tab} tab: '{label}' is not in SPEC"
                 for tab, rows in tables.items() for label in rows if (tab, label) not in in_spec]
    assert not problems, "PARAMETERS.md is out of step with params_io.SPEC:\n" + "\n".join(problems)
