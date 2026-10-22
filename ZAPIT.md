# Zapit

Opto, masking and ALM control sessions use Zapit, a scanning laser that shines light on chosen points of the skull on a share of trials; in these VGAT-ChR2 mice that silences the area underneath. Only rig 31108 has Zapit (31105 will). It runs on the rig's own PC, next to the control panel. For help, ask Quentin Pajot-Moric (see the README, [Contacts](README.md#contacts)).

**Parts of this page are still to be written** from the Zapit protocol. They are marked *To be written*: they are gaps, not steps you can skip.

## Before you start

- You need the laser induction before using Zapit. It is not covered here.
- Switch on the "laser on" light at the entrance door before you switch the laser on.

## Configurations

Zapit's stimulus configurations are in `Desktop\zapit_protocols` on 31108. Load the one that matches the mouse's preset:

| Preset | Configuration |
|---|---|
| `masking-regular`, `masking-hard-A`, `masking-hard-B` | `mask only.yml` |
| `opto-regular`, `opto-hard-A`, `opto-hard-B` | `ppc_stim.yml` |
| `control-alm-bi` | `alm_stim_bi.yml` |
| `control-alm-uni` | `alm_stim_uni.yml` |
| for testing | `test_stim.yml` |

- The presets set **Zapit Nb Conditions** to 1, and it must match the number of conditions in the loaded configuration. If Zapit has no configuration loaded, or the numbers differ, the session ends itself at the start (INSTRUCTIONS step 5).
- Stim Site and Stim Type in the panel are only labels. Where the laser actually goes is set by the configuration, so loading the wrong one gives data with the wrong labels, and nothing warns you.

## Starting Zapit

*To be written.*

## Calibration

- **Laser power:** every day or every week, depending on how much it drifts. A power meter's sensor goes under the laser, and its reading is compared with the expected value. *To be written: which meter, where its sensor goes, and the expected values.*
- **Aligning to the brain:** at the start of every Zapit session. *To be written.*

## Running a session

1. Switch on the laser light at the door.
2. Start Zapit, check the laser power if it is due, align, and load the configuration that matches the preset (see [Configurations](#configurations)).
3. In the panel, load the mouse and check Zapit Nb Conditions (INSTRUCTIONS steps 3–5), then launch.

*To be written: any further checks from the Zapit protocol.*

## After the session

*To be written:* closing Zapit and switching off the laser. Switch off the light at the door when you finish.

## When something goes wrong

- Turn Zapit off and on again.
- If the session ends by itself at the start, Zapit has no configuration loaded or Zapit Nb Conditions doesn't match it.
