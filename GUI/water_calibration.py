"""Water calibration: the numbers and the Bonsai command line behind the Calibrate button.

Kept apart from the window so the tests can run it off-rig. The procedure is HARDWARE.md, "Water": each
spout opens COUNT times at a valve time, the water is weighed, and the volume per opening decides
whether the rig's valve times in Params/Rigs.csv need changing.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import params_io

TARGET_UL = 5.0                   # µl per opening
RANGE_UL = (4.5, 5.5)             # a spout inside this needs no change
MAX_SPOUT_DIFFERENCE_UL = 0.5     # and the two spouts may differ by at most this
COUNT = 50                        # openings per run
RUNS_PER_SPOUT = 2
RUN_SPREAD = 0.10                 # a third run is offered when two runs differ by more than 10 %
SECONDS_PER_OPENING = 1.0         # the workflows open the valve once a second
START_SECONDS = 2.0               # a 1 s delay, then the first tick of the 1 s timer
SPOUTS = {'left': (1, 'Left_Valve_Time'), 'right': (2, 'Right_Valve_Time')}   # spout: (ValveNumber, Rigs.csv column)
LOG_COLUMNS = ['time', 'rig', 'spout', 'valve_ms', 'count', 'grams', 'ul_per_opening']


def ul_per_opening(grams: float, count: int = COUNT) -> float:
    """Volume per opening in µl from the weight of `count` openings in g (1 g of water = 1 ml)."""
    return grams * 1000.0 / count


def suggested_ms(valve_ms: float, ul: float) -> int:
    """The valve time that should give TARGET_UL, taking volume to be proportional to valve time."""
    return round(valve_ms * TARGET_UL / ul)


def in_range(ul: float) -> bool:
    return RANGE_UL[0] <= ul <= RANGE_UL[1]


def runs_disagree(grams: list) -> bool:
    """True when the runs of one spout differ by more than RUN_SPREAD of their mean."""
    if len(grams) < 2:
        return False
    mean = sum(grams) / len(grams)
    return mean > 0 and (max(grams) - min(grams)) / mean > RUN_SPREAD


def run_plan(runs_per_spout: int = RUNS_PER_SPOUT) -> list:
    """The order of the runs: left, right, left, right, ... as (spout, run number)."""
    return [(spout, k) for k in range(1, runs_per_spout + 1) for spout in SPOUTS]


def expected_seconds(count: int = COUNT) -> float:
    """How long the openings of one run take, once Bonsai has started."""
    return START_SECONDS + count * SECONDS_PER_OPENING


def workflow_for(rig_row: dict, repo: Path) -> Path:
    """The calibration workflow for this rig: the Arduino one if its valves run on the Arduino."""
    name = 'Manual_Calibration_Arduino.bonsai' if params_io.uses_arduino(rig_row) else 'Manual_Calibration_Harp.bonsai'
    return Path(repo) / 'Protocols' / 'Water_calibration' / name


def bonsai_command(bonsai_exe, workflow, valve: int, valve_ms: float, count: int = COUNT,
                   harp_port: str | None = None) -> list:
    """Bonsai, opened in its editor and started, as when the workflow is run by hand, with the workflow's
    settings given on the command line. The Harp workflow also takes the board's port; the Arduino one
    reads its port from Params/Rig_Params.csv. The editor stays open when the openings are done: the
    window's Done button closes it."""
    cmd = [str(bonsai_exe), str(workflow), '--start',
           f'-p:ValveNumber={valve}', f'-p:ValveTime={valve_ms:g}', f'-p:Count={count}']
    if harp_port:
        cmd.append(f'-p:PortName={harp_port}')
    return cmd


@dataclass
class Spout:
    """One spout's runs at one valve time."""
    name: str
    valve_ms: float
    grams: list = field(default_factory=list)
    count: int = COUNT

    @property
    def ul(self) -> float:
        """Mean volume per opening over the runs."""
        return ul_per_opening(sum(self.grams) / len(self.grams), self.count)

    @property
    def ok(self) -> bool:
        return in_range(self.ul)

    @property
    def suggested_ms(self) -> int:
        return suggested_ms(self.valve_ms, self.ul)


def spouts_close(left: Spout, right: Spout) -> bool:
    return abs(left.ul - right.ul) <= MAX_SPOUT_DIFFERENCE_UL


def new_valve_times(left: Spout, right: Spout) -> dict:
    """The valve times to put in Rigs.csv. A spout in range keeps its time, unless the two spouts are
    further apart than allowed, in which case both move to their suggested time."""
    close = spouts_close(left, right)
    return {SPOUTS[s.name][1]: (s.suggested_ms if not (s.ok and close) else round(s.valve_ms))
            for s in (left, right)}


def needs_change(left: Spout, right: Spout) -> bool:
    return not (left.ok and right.ok and spouts_close(left, right))


def log_run(path: Path, rig: str, spout: Spout, grams: float) -> None:
    """Append one run to the calibration log (a CSV, written with its header the first time)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec='seconds'), rig, spout.name, f'{spout.valve_ms:g}',
                         spout.count, f'{grams:g}', f'{ul_per_opening(grams, spout.count):.3f}'])
