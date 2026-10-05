# behaviour_rig

Operator tooling for running head-fixed mouse auditory-behaviour experiments across a room of training rigs. A Python control panel launches and manages the [Bonsai](https://bonsai-rx.org/) workflows that implement each task, handles per-rig and per-subject configuration, and provides the day-to-day utilities (camera, water flush/calibration, data transfer) an experimenter needs at the rig.

This is experimental-control software — lab instrumentation, not a data-analysis or machine-learning project. It is written to drive specific hardware on specific machines and is **not expected to run off a rig** or to be reused outside the lab; it is published as-is.

## Stack

**Operator GUI** — Python 3.10, `tkinter`

- `pandas` / `openpyxl` — reads session parameters from an Excel workbook
- `psutil` — finds and stops the running Bonsai process
- `python-osc` — OSC/UDP messaging with the running workflow
- `subprocess` — launches `Bonsai.exe <workflow> --start`

**Task engine** — [Bonsai](https://bonsai-rx.org/) (reactive, .NET). The `.bonsai` workflows under `Protocols/` implement the actual trial logic; the GUI starts and stops them.

**Custom Bonsai operators** — C# (`.cs`, built via `.csproj` + NuGet): an anti-bias transform (`Antibias`) and a TCP client for the Zapit optogenetic system.

**Microcontroller** — Arduino sketches (`.ino`) for digital I/O and PWM.

**Hardware assumed** — Harp behaviour board and/or Arduino over serial (COM ports), speaker and reward valves, a camera, and optional optogenetic inactivation (Zapit or fibre). Per-rig ports and calibration constants live in `Params/Rigs.csv`; each rig machine identifies itself via `C:\ProgramData\MouseRoom\rig.json`.

**Platform** — Windows (the rig PCs). Bonsai, the `.bat` launcher, and the `C:\...` / COM-port assumptions are Windows-only. A minimal macOS environment file is included, but it does not run the rig.

## Layout

- `GUI/` — the Python control panel (`Bonsai_GUI.py`), its conda environment, the launcher, and per-experimenter path configs
- `Protocols/` — Bonsai task workflows grouped by paradigm, plus Arduino firmware and calibration utilities
- `Params/` — the rig table (`Rigs.csv`), the session-parameter workbook, and the camera / flush workflows

## Protocols

Task families under `Protocols/` include amplitude-modulation discrimination, auditory (sound-categorisation) discrimination, delayed match-to-sample (DMTS), a staged PWM-auditory task (habituation → lick-to-release → full task, with anti-bias and optogenetic-inactivation variants), and a pro/anti task — alongside water- and sound-calibration workflows.

## Running it

Requires Bonsai installed per user and the `gui_env` conda environment. On a rig, `GUI/Launch_Bonsai_GUI.bat` activates the environment, syncs the repo to `origin/main`, and starts the control panel. Off a rig — without the hardware, COM ports, and `rig.json` — it will not run, though the workflows can still be opened in Bonsai for inspection.
