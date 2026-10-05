# Parameters

This file describes every parameter in the control panel, what it does in the `SOUND_CAT` task, and the rig settings in `Params/Rigs.csv`. It is generated from `GUI/params_io.py`, so labels, spreadsheet columns and options match the code. Where a parameter has no effect in the current workflow, it says so.

## How parameters get to the rig

1. Each mouse has a row in `Params/Mouse_Room_Params.xlsx` (sheet `Params`). **Load params** copies the selected mouse's row into the panel.
2. You can change values in the panel for this session; **Overwrite params** saves them. Nothing is written back to the spreadsheet.
3. Either button writes `Params/Subject_Params.csv`, one line of `Key: value,` pairs, which the workflow reads at start-up.
4. The rig's own settings come from its row in `Params/Rigs.csv` (written to `Params/Rig_Params.csv` when the panel opens).
5. Every parameter is saved in the session's `Session_Parameters` file, and most are repeated in every `Trial_Summary` row.

To change a mouse's defaults, edit the spreadsheet **in the repository** and push it. The rig launcher resets the code to the GitHub version every time it starts, so a spreadsheet edited on a rig is overwritten.

Yes/no parameters must be exactly `True` or `False`; a blank cell or `NaN` is refused when you press Load, because the workflow would wait for it forever.

Units: Sound Duration, Go Cue Duration, staircase steps and Min Stim Dur are in **milliseconds**; Inter Trial Interval, Timeout Duration and Response Window in **seconds**; Stim Range in **dB**.

## Setup tab

No task parameters: the Setup tab holds the **Experimenter** choice (used by Push Data) and the utility buttons, described in `INSTRUCTIONS.md`.

## Session tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| Subject | `Subject` | subjects in the spreadsheet |  | The mouse. Picks the spreadsheet row that **Load params** copies in, and names the data folder (`Data/<subject>/…`). **Push Data** only copies subjects whose ID starts with the experimenter's initials. |
| Protocol | `Protocol` | SOUND_CAT_DISC, SOUND_CAT_CONT, PRO_ANTI, SOUND_CAT |  | Which task runs. Use `SOUND_CAT` (sound categorisation) or `PRO_ANTI`. `SOUND_CAT_DISC` and `SOUND_CAT_CONT` are left over from an older version: the workflow has no branch for them, so no trials start. |
| Stage | `Stage` | Habituation, Lick_To_Release, Three_And_Three, Full_Task_Disc, Full_Task_Cont, Habituation_cont, Lick_To_Release_cont |  | Training stage within the protocol. Five stages have trial logic: `Habituation`, `Lick_To_Release`, `Three_And_Three` (the trial waits for a lick, which starts the sound), `Full_Task_Disc` (a fixed set of sound levels, see Nb Of Stim) and `Full_Task_Cont` (levels drawn from a distribution, see Distribution). `Habituation_cont` and `Lick_To_Release_cont` have no branch in the workflow, so no trials start. Only `Full_Task_Disc` and `Full_Task_Cont` record Window_Open_Time, First_Lick_Time and Early_Lick_Time. |
| Session Type | `Session_Type` | regular, opto, masking, washout, alm_control_uni, alm_control_bi |  | A label only: it is saved in Session_Parameters and in every Trial_Summary row but does not change what the rig does. For masking sessions it is the only record that the session was masking rather than opto (see Opto ON). |

## Stimulus tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| Distribution | `Distribution` | NaN, Uniform, Asym_Left, Asym_Right |  | `Full_Task_Cont` only. How each trial's sound level is drawn on the −1…+1 scale. `Uniform`: evenly across the whole range. `Asym_Left` / `Asym_Right`: one side's levels are drawn from an exponential concentrated near the category boundary, the other side's uniformly — not yet checked after the refactor (see README, Known issues). `NaN` is not valid for `Full_Task_Cont`. |
| Sound Duration | `Sound_Duration` | 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 | ms | Length of the white-noise sound. If Stim Dur Staircase is on, the workflow shortens it during the session. |
| Nb Of Stim | `Nb_Of_Stim` | NaN, 2, 4, 6, 8 |  | `Full_Task_Disc` only: how many distinct sound levels are used. |
| Stim Type | `Stim_Type` | NaN, PT, WN |  | Sound type. Only `WN` (white noise) is played by this workflow. |
| Stim Dur Staircase | `Stim_Dur_Staircase` | True, False |  | If `True`, Sound Duration is shortened during the session as performance improves. |
| Stim Dur Staircase Perf Thresh | `Stim_Dur_Staircase_Perf_Thresh` | NaN, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |  | Staircase: performance (fraction correct) above which the sound is shortened. |
| Stim Dur Staircase Step | `Stim_Dur_Staircase_Step` | NaN, 10, 20, 30, 40, 50 | ms | Staircase: how much each step shortens the sound. |
| Min Stim Dur | `Min_Stim_Dur` | NaN, 50, 100, 150, 200, 250, 300 | ms | Staircase: the sound is never shortened below this. |
| Stim Range Min | `Stim_Range_Min` | 40–99 in steps of 1 | dB | Sound level at the −1 end of the stimulus scale. |
| Stim Range Max | `Stim_Range_Max` | 40–999 in steps of 1 | dB | Sound level at the +1 end. Levels in between are linear, so 0 (the category boundary) is midway. The rig's speaker calibration in `Rigs.csv` converts dB to a sound-card setting. |

## Timing tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| Inter Trial Interval | `Inter_Trial_Interval` | 0–10 in steps of 1 | s | Wait between the end of one trial and the start of the next. |
| Timeout Duration | `Timeout_Duration` | 0–10 in steps of 1 | s | Extra wait after an incorrect response (no reward). |
| Response Window | `Response_Window` | 0–10 in steps of 1 | s | Time allowed to respond once the window opens. No lick in time is an abort: no reward and no timeout. |
| Go Cue Duration | `Go_Cue_Duration` | 0–1000 in steps of 50 | ms | Silent delay between the end of the sound and the opening of the response window. Despite the name, no go-cue sound is played. |

## Contingency tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| Rule | `Air_Puff_Contingency_Rule` | NaN, Pro_Only, Anti_Only, Blocks_30, Blocks_15, Random_Alternation |  | `PRO_ANTI` protocol only (air-puff rule). No effect in `SOUND_CAT`. |
| Show Contingency Switches | `Show_Contingency_Switches` | True, False |  | `PRO_ANTI` protocol only (marks contingency switches on the plots). No effect in `SOUND_CAT`. |
| Working Memory Type | `Working_Memory_Type` | NaN, Fixed, Variable |  | Not used by the current workflow (saved in Session_Parameters only). |
| Sound Air Puff Contingency | `Sound_Air_Puff_Contingency` | Low_Pro_High_Anti, Low_Anti_High_Pro |  | `PRO_ANTI` protocol only. No effect in `SOUND_CAT`. |
| Sound Contingency | `Sound_Contingency` | Low_Left_High_Right, Low_Right_High_Left |  | Which spout is correct. `Low_Left_High_Right`: quieter sounds (stimulus below 0) are left, louder sounds right. `Low_Right_High_Left`: the reverse. |

## Anti-bias tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| AntiBias | `AntiBias` | True, False |  | If `True`, the probability that the next trial is a right-side trial (recorded per trial as `P_Right`) is adjusted against the mouse's recent side bias. The three AB parameters below set how. |
| AB_Exp_Rate | `AntiBias_Exp_Rate` | NaN, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |  | Anti-bias: weighting of recent trials (exponential rate). |
| AB_Window | `AntiBias_Window` | NaN, 10, 20, 30, 40, 50 | trials | Anti-bias: how many recent trials are used to estimate the bias. |
| AB_Slope | `AntiBias_Sigmoid_Slope` | NaN, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |  | Anti-bias: steepness of the function that turns the bias estimate into P_Right. |
| Stable Start | `Stable_Start` | True, False |  | If `True`, the first Stable Start Window trials use easy sounds and limit same-side runs, before the normal stimulus rules take over. During these trials the Trial_Summary column `Correct_Side` is not updated (see README). |
| Stable Start Window | `Stable_Start_Window` | 10–50 in steps of 5 | trials | Length of the stable start. |
| Max Trials Consec | `Max_Trials_Consec` | 2–10 in steps of 1 | trials | Stable start: at most this many trials in a row on the same side. |
| Stable Stim Dist Boundary | `Stable_Stim_Dist_Boundary` | 0–0.9 in steps of 0.1 |  | Stable start: sounds are at least this far from the boundary on the −1…+1 scale (0.8 means only levels beyond ±0.8). |

## Opto tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| Opto ON | `Opto_ON` | True, False |  | Session-level switch for optogenetics and masking. When `True`: the masking light comes on at Onset_1 and goes off at Offset_1 on **every** trial, and on a random Perc_Opto_Trials % of trials the light source in Opto Type is triggered as well. Masking sessions are run as opto sessions with a Zapit configuration at 0 laser power; set Session Type to `masking` so the data say which it was. |
| % Trials | `Perc_Opto_Trials` | 0–105 in steps of 5 | % | Share of trials that get light. Drawn independently on each trial, so the exact number varies. |
| Light Freq (Hz) | `Light_Freq (Hz)` | 0–100 in steps of 10 | Hz | Not used by the current workflow (saved in Session_Parameters only). |
| Stim Site | `Stimulation_Site` | NaN, PPC, ACC, ALM |  | A label only (brain area), saved in Session_Parameters and every Trial_Summary row. Where the light actually goes is set in Zapit. |
| Stim Type | `Stimulation_Type` | NaN, Unilateral_Left, Unilateral_Right, Bilateral |  | A label only (unilateral/bilateral), saved like Stimulation_Site. |
| Opto Type | `Opto_Type` | NaN, Zapit, Fiber |  | Light source: `Zapit` (scanning laser, controlled over TCP) or `Fiber` (via the rig's Arduino Mega). |
| Zapit Nb Conditions | `Zapit_Nb_Conditions` | NaN, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |  | Must equal the number of conditions in the stimulus configuration loaded in Zapit. At start-up the workflow asks Zapit whether a configuration is loaded and how many conditions it has; if none is loaded or the number differs, the session ends itself. |

## Opto Timing tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| Onset_1 | `Opto_Onset_1` | Sound, Delay, Air_Puff, Go_Cue, Response_Window, Feedback, Reward, Timeout, Inter_Trial_Interval |  | Trial phase at which the light (and masking light) turns on. The phases that exist in `SOUND_CAT` are `Sound`, `Go_Cue`, `Response_Window`, `Feedback`, `Reward`, `Timeout` and `Inter_Trial_Interval`; `Delay` and `Air_Puff` belong to other protocols. |
| Onset_2 | `Opto_Onset_2` | Sound, Delay, Air_Puff, Go_Cue, Response_Window, Feedback, Reward, Timeout, Inter_Trial_Interval |  | Optional second turn-on phase (same choices as Onset_1). |
| Offset_1 | `Opto_Offset_1` | Sound, Delay, Air_Puff, Go_Cue, Response_Window, Feedback, Reward, Timeout, Inter_Trial_Interval |  | Trial phase at which the light from Onset_1 turns off. |
| Offset_2 | `Opto_Offset_2` | Sound, Delay, Air_Puff, Go_Cue, Response_Window, Feedback, Reward, Timeout, Inter_Trial_Interval |  | Trial phase at which the light from Onset_2 turns off. |
| Duration | `Opto_Duration` | 0–1000 in steps of 100 | ms | Not used by the current workflow (saved in Session_Parameters and Trial_Summary only). |

## Debug tab

| Panel label | Spreadsheet column | Options | Unit | What it does |
|---|---|---|---|---|
| Emulator | `Emulator` | True, False |  | If `True`, no hardware is used: the sound card, lick sensors and camera are skipped, no sound plays, and keys **1** and **2** on the keyboard act as left and right licks. For testing only. |
| Agent Sim | `Agent_Sim` | True, False |  | If `True`, a simulated mouse answers each trial (useful with Emulator). Its answers bypass the lick sensors. |
| Agent Performance | `Agent_Performance` | NaN, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |  | Simulated mouse: probability of answering correctly. |
| Agent Bias | `Agent_Bias` | NaN, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |  | Simulated mouse: bias towards the right spout. |
| Visualiser Window Size | `Visualiser_Window_Size` | 10–45 in steps of 5 | trials | Number of recent trials used by the on-screen performance plots. |

## Rig settings (`Params/Rigs.csv`)

One row per rig PC. The panel finds its row through `C:\ProgramData\MouseRoom\rig.json` (`{"rig": "<id>"}`) and refuses to launch if the row is missing or the Windows computer name doesn't match `Hostname`.

| Column | What it is |
|---|---|
| `rig` | Rig ID used in `rig.json`. |
| `Hostname` | Windows computer name of that rig's PC; a mismatch blocks launch (catches cloned disks). |
| `Room_ID, Rig_ID` | Labels, saved with the session. |
| `Arduino` | `TRUE`: lick sensors and valves run through an Arduino Nano on `Arduino_Port`. `FALSE`: a Harp behaviour board on `Harp_Beh_Port`. The panel converts `TRUE`/`FALSE` to the `True`/`False` the workflow expects. |
| `Harp_Beh_Port` | COM port of the Harp behaviour board (blank on Arduino rigs). |
| `Sound_Card_Port` | COM port of the Harp sound card. |
| `Arduino_Port` | COM port of the Arduino Nano (Arduino rigs). |
| `Arduino_Mega_Port` | COM port of the Arduino Mega that drives the masking light and fibre (blank if none). |
| `Left_Valve_Time, Right_Valve_Time` | Reward valve opening times in ms, from water calibration; each spout uses its own. (Before 5 Oct 2026 the right valve used Left_Valve_Time; see README, Known issues.) |
| `Speaker_Slope, Speaker_Y_Intercept` | Speaker calibration: sound-card setting = (dB − intercept) / slope. |

