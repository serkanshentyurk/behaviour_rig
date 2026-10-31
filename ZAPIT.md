# Zapit

Opto, masking and ALM control sessions use Zapit, a scanning laser that shines light on chosen points of the skull on a share of trials; in these VGAT-ChR2 mice that silences the area underneath. Only rig 31108 has Zapit (31105 will). It runs in MATLAB on the rig's own PC, next to the control panel. The [Zapit user guide](https://zapit.gitbook.io/user-guide/using-the-gui/introduction-using-the-gui) explains its GUI in detail. For help, ask Quentin Pajot-Moric (see the README, [Contacts](README.md#contacts)).

## Before you start

- You need the laser induction before using Zapit. It is not covered here.
- The mouse needs the Zapit surgery, with bregma and the 3 mm reference point marked on the skull ([SURGERY.md](SURGERY.md)).
- Switch on the "laser on" light at the entrance door before you switch the laser on.

## Configurations

A configuration file holds the stimulation settings: target coordinates, type of stimulation, laser power. They are in `Desktop\zapit_protocols` on 31108; make a new one in Zapit before you need it. Load the one that matches the mouse's preset:

| Preset | Configuration |
|---|---|
| `masking-regular`, `masking-hard-A`, `masking-hard-B` | `mask only.yml` |
| `opto-regular`, `opto-hard-A`, `opto-hard-B` | `ppc_stim.yml` |
| `control-alm-bi` | `alm_stim_bi.yml` |
| `control-alm-uni` | `alm_stim_uni.yml` |
| for testing | `test_stim.yml` |

- The presets set **Zapit Nb Conditions** to 1, and it must match the number of conditions in the loaded configuration. If Zapit has no configuration loaded, or the numbers differ, the session ends itself at the start (INSTRUCTIONS step 5).
- Stim Site and Stim Type in the panel are only labels. Where the laser actually goes is set by the configuration, so loading the wrong one gives data with the wrong labels, and nothing warns you.

## Checking the laser power

Every day or every week, depending on how much the power drifts.

1. Put the laser power meter's sensor in the laser's path.
2. Switch the laser on at a known power, for example 3 mW, and read the meter.
3. If the meter shows that power, carry on. If it doesn't, record the meter's reading at a few different settings, fit a straight line between the power you set and the power you measured, and use it to set the power you actually want.

## Starting a session

1. Switch on the laser light at the door.
2. Plug in the galvos' white power cord and switch on the laser. The NI device should already be on.
3. If the laser power check is due, do it now (above).
4. Open MATLAB and run `start_zapit` in the command window. The Zapit folder is already on the MATLAB path.
5. Head-fix the mouse on the rig.
6. In the Zapit GUI, check that the mouse's head is in view, and adjust the ROI if needed.
7. Click **Run Calibration**. Five or more calibration points (green circles) are enough. If fewer appear, increase the exposure and run it again.
8. On the **Calibrate Sample** tab, click **Calibrate Sample**, then click bregma and the 3 mm reference point. If you can't see the marks, shine a phone torch on the head, click Calibrate Sample (which freezes the camera image), then click the two points.
9. Choose **File › Load stim config file**, and load the configuration for the mouse's preset.
10. In the control panel, load the mouse, check Zapit Nb Conditions, and launch ([INSTRUCTIONS.md](INSTRUCTIONS.md) steps 3–6).

## During and after the session

- On every trial the laser and the masking light ramp down rather than switching off at once, which avoids rebound excitation in the neurons.
- Press End Session at a moment when the laser and the masking light are off.
- The camera may drop out if the PC is overloaded. That doesn't affect Zapit or the training.
- After the last Zapit session of the day, unplug the galvos' white power cord and switch off the laser, so it doesn't overheat. Then switch off the light at the door.

## When something goes wrong

- Turn Zapit off and on again.
- If the session ends by itself at the start, Zapit has no configuration loaded, or Zapit Nb Conditions doesn't match it.
- If fewer than five calibration points appear, increase the exposure and run the calibration again.
