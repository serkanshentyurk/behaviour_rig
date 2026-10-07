# behaviour_rig

Software for running head-fixed mouse auditory-behaviour experiments on the training rigs in room 311. A Python control panel reads each mouse's settings, starts the [Bonsai](https://bonsai-rx.org/) workflow that runs the task, and stops it at the end of the session. The panel also has the day-to-day utilities: camera preview, water-line flush and copying data to the server.

This is lab instrumentation. It drives specific hardware on specific Windows PCs and is not meant to run anywhere else.

| If you want to… | read |
|---|---|
| run a session (start here if you are new) | [INSTRUCTIONS.md](INSTRUCTIONS.md) |
| know what a parameter does | [PARAMETERS.md](PARAMETERS.md) |
| understand how the pieces fit, set up a rig, read the data files or change the code | this file |

## How a session runs

1. The control panel (`GUI/Bonsai_GUI.py`) reads the chosen mouse's row from `Params/Mouse_Room_Params.xlsx`. It also reads this PC's row from `Params/Rigs.csv` and writes both as one-line files, `Params/Subject_Params.csv` and `Params/Rig_Params.csv`.
2. **Launch Bonsai** starts Bonsai on `Protocols/Auditory_discrimination/Sound_Cat_V2.bonsai`, with `GUI/` as the working folder. All relative paths in the workflow (`..\Params\…`, `Extensions\…`, `../Data`) are relative to `GUI/`.
3. The workflow reads the two files, connects to the hardware, and runs trials until you press **End Session**. It writes six CSV files (plus camera video) to `Data/<subject>/<protocol>_<subject>_<date>/`.
4. **End Session** stops the trials, closes the valves and Zapit, and closes the data files about 2 s later. **Kill Bonsai** then closes Bonsai.
5. **Push Data** copies new session folders to the lab server.

A `SOUND_CAT` trial in the full task (`Full_Task_Cont` / `Full_Task_Disc`) goes:

1. Sound: white noise at a level set by the trial's stimulus value, for Sound_Duration.
2. Delay: a silent delay of Go_Cue_Duration (shown as "Delay before window" in the panel; the trial phase is still called `Go_Cue` in the data).
3. Response window: up to Response_Window seconds. If Go Cue Tone is on, a pure tone starts as the window opens. The first lick that **starts** after the window opens is the choice.
4. Outcome:
   - Correct: a reward from that spout's valve.
   - Incorrect: a timeout of Timeout_Duration.
   - No lick: an abort, with neither.
5. Inter-trial interval.

If Early Lick Abort is on, a new lick during the sound or the delay ends the trial at once: no window, no reward, the usual timeout, outcome `Early`.

## What's in the repository

```
GUI/
  Launch_Bonsai_GUI.bat   start here on a rig: activates gui_env, syncs the code, opens the panel
  Bonsai_GUI.py           the control panel (tkinter)
  params_io.py            the parameter table (SPEC) and the code that writes the parameter files
  GUI_env.yml             conda environment "gui_env" (mac_env.yml: macOS, cannot run a rig)
  Extensions/             Bonsai include files and zapit_TCPclient.cs, as used by the running workflow
  Extensions.csproj       needed for Bonsai to compile zapit_TCPclient.cs - do not delete
  paths/<name>.json       per-experimenter server paths used by Push Data
Params/
  Mouse_Room_Params.xlsx  one row per mouse (sheet "Params")
  Rigs.csv                one row per rig PC: ports, valve times, speaker calibration
  Camera.bonsai, Flush_Rig.bonsai   utility workflows behind the Camera and Flush Rig buttons
  Subject_Params.csv, Rig_Params.csv   written by the panel; not in git
Protocols/
  Auditory_discrimination/Sound_Cat_V2.bonsai   the task workflow the panel runs
  Auditory_discrimination/Sound_Cat_V2.bonsai.layout   its window layout - keep it next to the workflow
  Arduino/                firmware for the Arduino lick/valve board
  …                       other task families, calibration workflows and sound files
tests/test_rig_consistency.py   checks the GUI, parameter files and workflow agree
Data/                     session data, written by Bonsai; not in git
```

**Stack.**
- The panel is Python 3.10 with tkinter, pandas/openpyxl, psutil and python-osc. It talks to the running workflow over OSC on port 1334.
- The task runs in Bonsai 2.7, which saves the workflow as version 2.7.0.
- Custom operators are written in C#, and the lick/valve board is an Arduino.
- Hardware: Harp sound card, plus a Harp behaviour board or an Arduino Nano for lick sensors and valves, camera, and optionally Zapit or fibre optogenetics with an Arduino Mega for the masking light.

## Setting up a rig PC (once)

1. **Bonsai 2.7** installed for the Windows user, so that `%LOCALAPPDATA%\Bonsai\Bonsai.exe` exists. It needs these packages:
   - Bonsai.Core, Bonsai.System, Bonsai.Design.Visualizers
   - Bonsai.Scripting.Expressions, Bonsai.Scripting.IronPython
   - Bonsai.Numerics, Bonsai.Dsp, Bonsai.Vision, Bonsai.Osc, Bonsai.Windows.Input, Bonsai.Shaders
   - Bonsai.Harp, Bonsai.Harp.CF
   - BonVision

   This list is taken from the workflow file's namespace declarations. Opening the workflow once in the editor shows any that are missing.
2. **Miniconda or Anaconda** in the user folder (`%USERPROFILE%\miniconda3` or `anaconda3`; the launcher looks there). Then run `conda env create -f GUI/GUI_env.yml` to create `gui_env`.
3. **Git** on the PATH, and the repository cloned on the PC.
4. **Register the rig.**
   - Add a row to `Params/Rigs.csv`, with the PC's Windows computer name in `Hostname`, then commit and push.
   - On the PC, create `C:\ProgramData\MouseRoom\rig.json` containing `{"rig": "<id>"}`.
5. **A desktop shortcut** to `GUI\Launch_Bonsai_GUI.bat`.
6. **Check.** Start the panel; the bottom line should read `Rig <id>   (<computer name>)` in green.

## Output files

Each session writes into `Data/<subject>/<protocol>_<subject>_<YYYY>_<M>_<D>/`, for example `Data/SS14/SOUND_CAT_SS14_2026_10_5/`. Every file name ends with the session's start time, e.g. `Trial_Summary2026-10-05T15_16_16.csv`, so several sessions on one day share a folder.

All `Time` columns are **seconds since midnight on the rig PC's clock**, and every file uses the same clock, so they can be aligned directly.

| File | One row per | Columns |
|---|---|---|
| `Trial_Summary` | completed trial | 54 columns: trial result, timing, and the session settings repeated on every row |
| `Session_Parameters` | session (one data row) | every parameter, as name, value, name, value … |
| `Trial_Epochs` | change of trial phase | `Epoch`, `Time` |
| `Long_Form_Timestamps` | event | `Label`, `Data`, `Time` |
| `Detected_Licks` | change of a lick sensor | `State`, `Time`, `Spout` |
| `Zapit_Timestamps` | message to or from Zapit | `Label`, `Data`, `Time` |
| `Video_data…`, `Video_timestamps…` | camera frame | camera video and frame times |

### Reading Trial_Summary

A row is written when a trial ends. A trial cut short by End Session has no row, although its events are still in the other files.

The main columns:
- **Trial and stimulus.** `Trial_Number` and `Stim_Relative` (the stimulus on a −1…+1 scale; 0 is the category boundary).
- **Response.**
  - `First_Lick` (Left or Right), `Correct` and `Abort_Trial`.
  - `Trial_Outcome`: Correct, Incorrect, Abort (no lick in the window) or Early (ended by an early lick, when Early Lick Abort is on). Early trials also have `Abort_Trial` True, and blank `Window_Open_Time`, `First_Lick_Time` and `Response_Latency`.
  - `Reward_Side`.
- **Timing.**
  - `Window_Open_Time` and `First_Lick_Time`, plus `Response_Latency` in ms.
  - `Early_Lick_Time`: the first new lick between the sound and the window opening. The trial carries on regardless.
  - `Trial_End_Time`.
- **Labels.** `Session_Type`, `Stimulation_Site` and `Stimulation_Type`.
- **Repeated session settings.** Durations appear as h:mm:ss text, so `00:00:00.3000000` is 300 ms.

Things to know when analysing it:
- **Response_Latency changed meaning on 5 October 2026.**
  - Files that have a `Window_Open_Time` column use the new meaning: (First_Lick_Time − Window_Open_Time) × 1000, blank on aborts.
  - Older files used a timer that counts in screen frames. It reads 0 when a lick was already in progress as the window opened, and on abort rows it holds the previous trial's value.
- **Which licks count changed on the same date.** From then on, only a lick that starts after the window opens counts. Before, a contact already in progress when the window opened was taken as the choice.
- **On abort rows,** ignore `First_Lick`, `Correct` and `Reward_Side`: they keep the previous trial's values.
- **`Reward_Side`** only means something when `Correct` is True.
- **`Correct_Side` is not updated during the stable start.** That's the first Stable_Start_Window trials when Stable_Start is on. Work the correct side out from `Stim_Relative` and `Sound_Contingency` instead: with `Low_Left_High_Right`, a negative value means left.
- **Ignore trial 1.** Its timing is unreliable because Bonsai is still starting up.

### The other files

- **Session_Parameters.**
  - Ignore the header row: it's Bonsai's automatic naming and has duplicates. The data row alternates name and value.
  - In Python: `row = list(csv.reader(open(path)))[1]; params = dict(zip(row[0::2], row[1::2]))`.
- **Trial_Epochs.**
  - A trial runs `Sound`, `Go_Cue`, `Response_Window`, then `Feedback` followed by `Reward` or `Timeout`, then `Inter_Trial_Interval`. Aborts go straight from the response window to the interval.
  - An `Early` trial goes from `Sound` (or `Go_Cue`) straight to `Feedback` and `Timeout`, so opto light set to switch off at `Feedback` still does.
  - The first row is a blank start-up value.
- **Long_Form_Timestamps.**
  - `Sound`: the stimulus value, at the moment the sound is triggered.
  - `First_Lick`: Left or Right.
  - `Left_Valve`, `Right_Valve`: True when a valve opens, False when it closes.
  - The first rows are start-up values: a blank `First_Lick` and `Sound` 0.0.
- **Detected_Licks.**
  - `Beam1Bool` is the left spout and `Beam2Bool` the right. `False` means the beam was broken (a lick starts) and `True` means it was released.
  - It is not a clean list of licks. Both sensors start as `False` when the session starts, and the workflow writes `True` at trial starts. Count licks as `False` rows, skipping the first two.
  - On Arduino rigs the time is when Bonsai received the event, about 25 ms after the lick.
- **Zapit_Timestamps.**
  - Each row is a command to Zapit or Zapit's reply.
  - The `Data` field is a tuple such as `(254, 0, 0, 0, 0, 0, 0)`. It contains commas and isn't quoted, so a normal CSV reader splits it. Read the file line by line instead.

## Changing the code or the workflow

- **Change things on a development machine, then commit and push.** Every rig resets its copy to `origin/main` each time the launcher starts, so changes made on a rig are lost. That includes the spreadsheet.
- **Open the workflow with the panel's Edit Workflow button**, or start Bonsai with `GUI/` as the working folder. Started from the Start menu, the relative paths don't resolve.
- **Keep `Sound_Cat_V2.bonsai.layout` next to the workflow, under the same name.**
  - It holds the windows that open during a session, and Bonsai rewrites it whenever you save.
  - If you replace the workflow file in some other way, replace the layout with its matching file, or the windows won't open.
  - Don't commit an empty `.bonsai/Settings/Sound_Cat_V2.layout`: Bonsai 2.8 and later would use it instead of the layout above.
- **Keep `GUI/Extensions.csproj`.** Without it, Bonsai doesn't compile `GUI/Extensions/zapit_TCPclient.cs`. The workflow then fails with "proxy for the unknown type zapit_TCPclient" and won't start, even in emulator mode.
- **Two `Extensions/` folders exist.** The running workflow uses `GUI/Extensions/`, because paths are relative to `GUI/`. `Protocols/Auditory_discrimination/Extensions/` is a copy; if you change one, change both.
- **To add a parameter**, you need three things: a row in `SPEC` in `GUI/params_io.py`, a column in the spreadsheet, and a parser in the workflow's `Params` group. The tests fail until all three match.
- **Run the tests before pushing.**

## Tests

From the repository root, in any environment with pandas, numpy, openpyxl and pytest (gui_env doesn't include pytest):

```
python -m pytest tests
```

They read the workflow file directly, with no Bonsai or hardware, and check that:
- every value the workflow reads is written by the panel, for every mouse and rig, and that every panel parameter is read;
- every Trial_Summary column points at the right value;
- every data file is closed when a session ends;
- `GUI/Extensions.csproj` is present;
- in `SOUND_CAT`, each spout's valve is timed by its own calibration;
- in `Full_Task_Cont`, the stimulus stage ends only after the sound has played;
- the early-lick abort cuts the trial before the window and ends with the timeout and outcome `Early`;
- the go-cue tone sets its level, plays at the window opening and stops after its duration;
- no group's connections form a loop (Bonsai can't build or open a group with one);
- the window layout lines up with the workflow;
- the response, timing and latency rules described above are wired as intended.

They can't tell you whether Bonsai builds and runs the workflow. For that, open it in Bonsai and run a short emulator session.

## Known issues

These are current as of 5 October 2026.

- **Right valve time, fixed on 5 October 2026 for `SOUND_CAT`.** Before that, every right reward opened the right valve for `Left_Valve_Time`. On 31104 and 31105 the right spout was open about 26% longer than its calibration; on the other rigs the difference was under 5%. Each session's `Session_Parameters` records its `Rig_ID`. `PRO_ANTI` still opens the right valve for `Left_Valve_Time`; that is left for the protocol's owner to change.
- **Trial 1 timing is unreliable.** Ignore trial 1 in analysis.
- **`Correct_Side` isn't updated during the stable start** (see Reading Trial_Summary).
- **Some options start no trials.** Protocol `SOUND_CAT_DISC` and `SOUND_CAT_CONT`, and Stage `Habituation_cont` and `Lick_To_Release_cont`, are offered by the panel but have no trial logic.
- **Asymmetric distributions, fixed on 5 October 2026.** Before that, with `Asym_Left` or `Asym_Right` the stimulus stage ended as soon as the side was chosen. The go-cue delay started at sound onset, so the window opened as the sound ended, and some trials played no sound at all; those trials were scored, and rewarded, against the previous trial's stimulus. Sessions run with these distributions before that date are affected. `Asym_Left` was checked in the emulator before the fix; check both after it.
- **The "fresh licks only" rule hasn't been tested on hardware** with a contact held through the window opening.
- **Trial timing follows the display refresh.** The sound and go-cue delays vary by about one frame, about 17 ms. The times recorded in the files are exact.
- **The go-cue tone is optional and off by default.**
  - Its level is only approximate, because the speaker calibration was made with white noise.
  - With the tone on, keep Delay before window at 50 ms or more. The white noise is ended by starting sound slot 30 on the card, and if that lands just after the tone starts it replaces the tone.
  - The tone isn't logged separately; it starts at `Window_Open_Time`.
  - It can't be heard in emulator mode, because there's no sound card.
- **Lick times lag on Arduino rigs** by about 25 ms, because the serial link runs at 9600 baud.
- **Masking and opto sessions look identical** in the data except for `Session_Type`. The laser power is set in Zapit.
