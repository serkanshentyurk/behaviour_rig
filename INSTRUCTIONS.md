# Running a session

A step-by-step guide for someone who has not used the rig before. It covers the software only; water restriction, preparing the rig, cleaning and moving a mouse through training are in [TRAINING.md](TRAINING.md), the rigs themselves in [HARDWARE.md](HARDWARE.md), and Zapit in [ZAPIT.md](ZAPIT.md). Parameter meanings are in [PARAMETERS.md](PARAMETERS.md); how the system works is in [README.md](README.md).

If something here disagrees with what you see on the rig, trust the rig and tell whoever maintains the repository.

## The control panel at a glance

The panel has tabs along the top: **Setup**, **Session**, **Stimulus**, **Timing**, **Contingency**, **Anti-bias**, **Opto**, **Opto Timing**, **Debug**. Each tab except Setup holds part of the mouse's settings.

The bar at the bottom is visible from every tab:

| Button | What it does |
|---|---|
| **Load params** | copies the selected mouse's settings from the spreadsheet into all tabs, and saves them for Bonsai |
| **Overwrite params** | saves the values currently in the tabs for Bonsai (use after changing anything) |
| **Launch Bonsai** | starts the session; while a session runs, the same button reads **End Session** |
| **Kill Bonsai** | closes Bonsai |

Under the buttons are three lines of text:
- **The rig line** (for example `Rig 31104 (DESKTOP-L4E2IPB)`, in green) says which rig the panel thinks it is.
- **The preset line** appears after Load: the mouse, its preset (or custom), and any field you have changed since loading.
- **The status line** says what to do next:

| Status line | Meaning |
|---|---|
| Load or Overwrite params to begin | nothing saved yet |
| Ready to launch | settings saved; you can launch |
| Unsaved changes - press Overwrite | you changed a field since the last save |
| Bonsai running | a session is running |
| Session ended - press Kill Bonsai | End Session was pressed; Bonsai is still open |
| Closing data files... | Kill Bonsai is waiting for the data to be written |
| Copying data to the server... | Push Data is running |

Button colours:
- **Green:** ready or done.
- **Amber:** needs attention (for example, Overwrite after a change).
- **Red:** running, or the stop button.
- **Grey:** not ready.

Field colours: after Load, blue fields come from the mouse's preset and yellow fields from its own row in the spreadsheet. A field you change turns orange.

## Before you start

- Close any Bonsai windows left open from earlier.
- The sound card, lick-sensor board (Harp behaviour board or Arduino) and camera should be connected and powered.
- For opto or masking sessions, Zapit must be running with the right stimulus configuration loaded (see step 5).

## Step by step

1. **Start the panel.** Double-click the desktop shortcut, or `GUI\Launch_Bonsai_GUI.bat`.
   - A black window opens first. It updates the code from GitHub and then opens the panel. Leave it open: closing it closes the panel, and it shows error messages if something goes wrong.
   - Check the rig line at the bottom. If it says **RIG NOT CONFIGURED - launch disabled** in red, stop and see Troubleshooting.

2. **Choose the experimenter** on the **Setup** tab (your initials; the list comes from the files in `GUI\paths`). This is only used to decide where Push Data copies to.

3. **Choose the mouse** on the **Session** tab (Subject), then press **Load params**.
   - The mouse's values fill every tab and you get "Params successfully loaded", which says whether a preset was used. If the mouse's Preset is not `custom`, the values its preset sets come from `Params\Presets.csv` (blue fields) and the rest from its row (yellow). If its row holds values the preset replaces, the message lists them.
   - Load params and Launch Bonsai turn green and the status line reads "Ready to launch".
   - If Load refuses with "Not loaded - fix the spreadsheet", a value in the spreadsheet or in `Presets.csv` is unusable (see Troubleshooting).

4. **Check the settings.** At least look at:
   - **Session:** Stage and Session Type. Session Type is the only record of what kind of session this was.
   - **Opto:** Opto ON, Opto Type, Stim Site and Stim Type.
   - **Timing:** the durations, and **Early Lick Abort** (ends the trial if the mouse licks before the window opens; off unless the spreadsheet or you switch it on).
   - **Debug:** Emulator and Agent Sim must both be **False** for a real session.

   If you change anything, the status line says "Unsaved changes - press Overwrite". Press **Overwrite params** before launching; Launch refuses otherwise. Overwrite lists the fields you changed and asks you to confirm. If you changed a value the mouse's preset sets but left Session Type as the preset's name, it says so, because the data will be labelled with that preset.

   Changes made here apply to **this session only** and are not written back to the spreadsheet. To change a mouse's defaults, the spreadsheet `Params/Mouse_Room_Params.xlsx` has to be edited in the repository and pushed. An edit made on the rig is overwritten the next time the launcher starts. To move a mouse onto a preset, or back to `custom`, change its Preset cell in the spreadsheet; to change what a preset sets for every mouse on it, edit `Params/Presets.csv` the same way.

5. **Opto and masking only: set up Zapit before launching.** Starting Zapit, its configurations and the laser light at the door are in [ZAPIT.md](ZAPIT.md).
   - **Zapit Nb Conditions** (Opto tab) must equal the number of conditions in the configuration loaded in Zapit. At start-up the session checks this; if Zapit has no configuration loaded or the number differs, the session ends itself.
   - **Masking sessions:** use one of the `masking-…` presets. They run as opto sessions (Opto ON = True, Opto Type = Zapit), so load a Zapit configuration at 0 laser power. The data can only tell masking from opto by the Session Type the preset sets.

6. **Press Launch Bonsai.**
   - Bonsai opens and starts the task. Within a few seconds four windows appear: the camera image, a text window with the latest trial's summary, a rolling performance graph and a bar graph. Trials start on their own.
   - The Launch button turns red and reads **End Session**.
   - While the session runs, don't click Stop in Bonsai, don't close Bonsai's windows, and don't edit the workflow.
   - Launch refuses while any Bonsai is open (Camera, Flush Rig, the editor, or a session not yet killed); press Kill Bonsai first.

7. **Ending the session.**
   - Press **End Session**. The trial in progress is abandoned (it won't appear in Trial_Summary), the valves close, Zapit is told to stop, and the data files are closed about 2 seconds later.
   - The status line then says "Session ended - press Kill Bonsai". Press **Kill Bonsai** to close Bonsai.
   - If you press Kill Bonsai without End Session, it sends End Session itself and waits about 4 seconds before closing Bonsai.
   - **Never close Bonsai any other way during a session** (window X, Task Manager, shutting the PC down). The end of every data file is lost.
   - Always Kill Bonsai before launching another session.

8. **Copy the data to the server.** On the Setup tab, press **Push Data**.
   - It copies session folders from `Data\` on this PC to your folder on the server (the path is in your file in `GUI\paths`).
   - It only copies mice whose ID starts with your initials (SS…, QP…).
   - It never overwrites a file already on the server and never deletes anything on the rig. At the end it says how many files it copied. A file already on the server with a different size (usually copied from an unfinished session) is listed and left alone, so you can check it by hand.
   - It refuses while Bonsai is open, so that no file is copied while it is still being written.
   - The server must be mapped as a drive in Windows (Z: on the rig PCs; see the README, [Access you need](README.md#access-you-need)).

9. **Check the data** if you want: `Data\<mouse>\<protocol>_<mouse>_<date>\` should contain six CSV files whose names end with the session's start time. The README explains each file.

## Other buttons (Setup tab)

| Button | What it does |
|---|---|
| **Flush Rig** | starts the water-line flushing workflow (`Params\Flush_Rig.bonsai` on Arduino rigs, `Params\Flush_Rig_Harp.bonsai` on Harp rigs); press again to stop it |
| **Camera** | starts a live camera preview (`Params\Camera.bonsai`); press again to stop it. Stop it before launching a session, which uses the camera itself |
| **Edit Workflow** | opens the task workflow in the Bonsai editor, set up so its file paths work; for looking at or changing the workflow (read the README first) |
| **Test Speakers**, **Calibrate** | not implemented (greyed out) |

**Kill Bonsai** closes every Bonsai window on the PC, including an editor opened with Edit Workflow. Save any work in the editor first.

## Trying it out without a mouse (emulator)

Use a test subject (for example `TEST`), so the test data don't end up in a real mouse's folder.

1. Load the test subject's params.
2. On the **Debug** tab, set **Emulator = True**. No hardware is used and no sound plays.
3. Choose how responses are made:
   - **Agent Sim = False:** you respond yourself. Key **1** is a left lick and key **2** a right lick.
   - **Agent Sim = True:** a simulated mouse answers, with the accuracy and bias set by Agent Performance and Agent Bias.
4. Set **Opto ON = False** unless Zapit is running. Otherwise the session talks to Zapit and may end itself (step 5 above).
5. Press Overwrite params, then Launch Bonsai.
6. End and kill as in step 7.

The data are written exactly as in a real session. Ignore the first trial of any session, emulated or not: its timing is unreliable.

To try **Early Lick Abort** in the emulator: switch it on, press 1 or 2 while the sound or delay is running, and check that the trial ends with outcome `Early` after the timeout.

## Troubleshooting

| What you see | Likely cause | What to do |
|---|---|---|
| Red **RIG NOT CONFIGURED - launch disabled**, with a message | this PC isn't registered: `C:\ProgramData\MouseRoom\rig.json` is missing, names a rig that isn't in `Params\Rigs.csv`, or the computer name doesn't match that row | the message says which; register the rig as in the README |
| "Select a subject" | no mouse chosen | choose one on the Session tab |
| "No params available for this subject" | the mouse isn't in the spreadsheet | add a row to the spreadsheet (in the repository) |
| "Not loaded - fix the spreadsheet: … must be True or False" | a yes/no column is empty, `NaN`, or another value | fix that cell in the spreadsheet; for this session you can set the field by hand and press Overwrite |
| "Spreadsheet is missing columns: …" | the spreadsheet is older than the panel | add the listed columns |
| "Not loaded - fix the spreadsheet: … is not one of: …", "… is not a number" or "… is empty" | a value is misspelt, not a number, or missing where one is needed | fix that cell in the spreadsheet, or in `Presets.csv` if the message names a preset |
| "Preset '…' is not in Presets.csv" | the mouse's Preset cell names a preset that doesn't exist | correct the Preset cell, or add the preset to `Params\Presets.csv` |
| "Subject was changed after loading …" | another subject was picked after Load | press Load params for the subject you want |
| "Bonsai is already open … Press Kill Bonsai first" | Camera, Flush Rig, the editor or an earlier session is still open | press Kill Bonsai, then try again |
| "All params must be filled in" | a field still says "Select" | fill it in, then Overwrite |
| "Protocol can't launch without params!" | nothing loaded or saved yet | press Load params (or fill everything in and press Overwrite) |
| "Params changed since last save - press Overwrite first" | a field changed after the last save | press Overwrite params |
| "Bonsai.exe not found at …" | Bonsai isn't installed for this Windows user | install Bonsai (README, rig setup) |
| Bonsai shows "proxy for the unknown type zapit_TCPclient" and won't start | `GUI\Extensions.csproj` is missing | restore it from git; Bonsai needs it to compile the Zapit client |
| Bonsai opens but no trials ever start | with Zapit, no configuration is loaded or Zapit Nb Conditions doesn't match | check those settings; for anything else, note any red node and its message in Bonsai |
| The windows (graphs, camera, trial text) don't appear | the layout file next to the workflow is missing or doesn't match it | see "Changing the code or the workflow" in the README |
| Data files stop mid-line or miss the last minutes | Bonsai was closed without End Session | always End Session, then Kill Bonsai |
| "Select an experimenter before pushing" | no experimenter chosen | choose one on the Setup tab |
| "No server found on current machine" | the server drive isn't mapped, or the path in `GUI\paths\<name>.json` doesn't exist on it | map the drive and try again |
| Push Data lists files "on the server with a different size" | an earlier push copied a session before it had finished | compare the two copies by hand; Push Data never overwrites |
| No water comes out during Flush Rig or a session, on a Harp rig (31101, 31102) | the board's valve outputs have switched to pulse mode | in the Harp Behaviour app, set Use pulse to false ([HARDWARE.md](HARDWARE.md#the-harp-behaviour-app)) |
| Bonsai won't start, with an error about a COM or serial port | a USB cable or hub was swapped and the ports changed | find the new ports and update `Rigs.csv` ([HARDWARE.md](HARDWARE.md#com-ports)) |
