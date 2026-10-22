# The rigs

What is on the rigs in room 311, how to set up a rig PC, calibration, and what to do when something stops working. Ports and calibration values are in `Params/Rigs.csv`, described in [PARAMETERS.md](PARAMETERS.md#rig-settings-paramsrigscsv). For anything not covered here, ask Quentin Pajot-Moric (see the README, [Contacts](README.md#contacts)).

## What each rig has

Every rig has:
- a PC in the room, with one account; the password is the user name;
- a Harp sound card and a speaker;
- two lick spouts, each with a touch sensor and a water valve, fed from a 20 ml syringe. Each sensor has a light that comes on when the spout is touched;
- a board for the sensors and valves: a Harp behaviour board or an Arduino Nano;
- a camera.

Left and right always mean the mouse's left and right.

| Rig | Sensors and valves | Optogenetics |
|---|---|---|
| 31101 | Harp behaviour board | fibre, with masking light |
| 31102 | Harp behaviour board | fibre, with masking light |
| 31104 | Arduino | none |
| 31105 | Arduino | none yet: Zapit is planned and the equipment is ready |
| 31107 | Arduino | none |
| 31108 | Arduino | Zapit, with masking light |
| 31109 | Arduino | none |

The masking light, and the fibre light on 31101 and 31102, are driven by a second Arduino (an Arduino Mega); Zapit takes its commands from the task workflow directly. Only the opto rigs have a masking light, so only they can run masking sessions. The masking light needs no settings.

Everything stays switched on; there is no start-up order.

## The spouts

The spouts sit on a mover (Thorlabs) with knobs for fine adjustment. Set them slightly below the mouse's mouth, the same distance from both sides, so it can reach and lick both. Touching a spout turns on its sensor light, which is the quickest way to check that a sensor works.

## Water

Use regular water. Fill the syringe to the top (20 ml) every morning and keep it above 15 ml: the valves are calibrated with a full syringe, and the reward gets smaller as the level drops.

If air stays in the tubes, keep flushing until it is gone, with Flush Rig or, on the Harp rigs, by hand in the Harp Behaviour app (below).

When and how to change the tubing or a valve, and where spares and tools are kept, isn't written down yet: ask Quentin.

## COM ports

Each board appears in Windows as a COM port, and the ports are recorded in the rig's row of `Params/Rigs.csv`. The numbers can change when a USB cable or hub is swapped (not when the PC restarts), and Bonsai then won't start. Find the new ports in Windows Device Manager or the Harp Behaviour app, update `Rigs.csv` in the repository, and push.

## The Harp Behaviour app

The rigs with a Harp behaviour board (31101, 31102) have the **Harp Behaviour** app on the desktop. Open it and connect to the board's COM port (`Harp_Beh_Port` in `Rigs.csv`). Close it again before launching a session or Flush Rig: only one program can use the port at a time.

**When the valves don't open.** Sometimes the board switches its outputs to pulse mode by itself; why is not known. The valves then open and close at once, so little or no water comes out, during Flush Rig or a session. To fix it, go to the **Output pulse** tab and set **Use pulse** to false for the rows Port 0/1/2 12 V and Poke 0/1/2 Valve (ms).

**Opening the valves by hand.** On the **Interface** tab, tick Port 1 and Port 2 (the 12 V outputs) and send; untick them and send again to close the valves. This is also a way to flush by hand.

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

## Calibration

Water calibration sets each valve's opening time (`Left_Valve_Time`, `Right_Valve_Time`); sound calibration sets `Speaker_Slope` and `Speaker_Y_Intercept`. Both go into the rig's row of `Params/Rigs.csv`, as a change in the repository. Calibrate every week, with a full syringe.

**The procedures are not written yet.** They will come from the water and sound calibration protocols. What exists so far:

- Water: `Protocols/Water_calibration/Manual_Calibration_Arduino.bonsai` (Arduino rigs) or `Manual_Calibration_Harp.bonsai` (Harp rigs), with `Water_Calib_Plots.ipynb`.
- Sound: `Protocols/Sound_Calibration/WN_Volume_Calibration.bonsai` and `Test_Volume_Cal.bonsai`, with `Vol_Calib_Plots.ipynb`.

These workflows predate the current folder layout, and the panel does not run them. Check their file paths and ports in the editor before using one: the sound workflows read `Rig_Params.csv` from `C:\Users\MouseRoom\Desktop\HeadFixedBehavior\Params\` or `C:\Users\Psychophysics\Desktop\HeadFixedBehavior\Params\`, `Manual_Calibration_Arduino.bonsai` reads `../../Params/Rig_Params.csv`, and `Manual_Calibration_Harp.bonsai` has `COM4` written in.

## When something stops working

| Problem | What to do |
|---|---|
| Bonsai won't start, with an error about a COM or serial port | the ports have probably changed: see [COM ports](#com-ports) |
| No water during Flush Rig or a session, on a Harp rig | the pulse setting: see [The Harp Behaviour app](#the-harp-behaviour-app) |
| Air in the tubes | keep flushing until it is gone |
| The sound card stops responding | switch it off and on again (see the README, Known issues, on the go-cue tone) |
| Zapit misbehaves | see [ZAPIT.md](ZAPIT.md#when-something-goes-wrong) |
| A leaking or stuck valve, worn tubing, a sensor that doesn't respond to touch, a board to replace | not written down yet: ask Quentin |
