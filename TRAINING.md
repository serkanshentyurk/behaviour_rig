# Training a mouse

What an experimenter does with a mouse, from surgery to the full task, once the rigs are set up and working. For how to use the control panel itself, see [INSTRUCTIONS.md](INSTRUCTIONS.md); this page points to its steps instead of repeating them.

## Records

Every mouse on water restriction goes in **Dammy Mouse Restriction.xlsx**, on SharePoint under SWC PIL › restriction sheets › Akrami Lab.

- Once per mouse: cage ID, mouse ID, mouse name (the Subject used on the rigs, e.g. SS14) and start weight.
- Every day the mouse is on restriction: its weight, and whether it was trained or given water.

## Adding a mouse

Do this in the repository on your own computer, before the mouse's first session. Not on a rig: each rig resets to the repository's main branch whenever its launcher starts, so an edit made on a rig is lost.

1. Add a row for the mouse to `Params/Mouse_Room_Params.xlsx` (sheet `Params`) and fill in every column.
2. Start its name with your initials (SS…, QP…). Push Data only copies mice whose name starts with the experimenter's initials, so a mouse named otherwise never reaches the server.
3. Set its **Preset** to `habituation`, the first training step (see [Moving through training](#moving-through-training)), or to `custom`.
4. From the repository root, run `python -m pytest tests`; the tests check that every mouse loads.
5. Commit and push. Each rig picks the change up the next time its launcher starts.

## From surgery to the first session

About a week (5–7 days).

1. **Handling, the 3 days after surgery.** Put your hand in the cage and let the mouse sniff it and walk on it, then start picking it up and handling it.
2. **Water restriction and the rig, 1–2 days.** Weigh the mouse and record its start weight, then start water restriction (see [Water restriction](#water-restriction)). Keep handling it, and give it water from your hand. Put it on the rig without head-fixing it, let it explore, and give it water there.
3. **Head-fixing, 1–2 days.** As in step 2, and now also head-fix the mouse and give it water while it is head-fixed.

Then start training with `habituation`.

## Water restriction

- Weigh every mouse every day it is on restriction, before any session, and record the weight.
- Its weight must not go below 80% of the start weight. If it does, give it more water.
- On days without training, give each mouse 1 minute with the water bottle in its cage.
- After a session in which a mouse drank little, top it up from the bottle, for less than a minute depending on how much it drank on the rig.

## Every week

- Calibrate water and sound on each rig (see [Calibration](#calibration)).

## Every day

### In the morning, on each rig

1. Fill the water syringe to the top.
2. Start the panel ([INSTRUCTIONS.md](INSTRUCTIONS.md) step 1).
3. **Flush.** On the Setup tab press **Flush Rig**. Each time it starts, it opens both valves for about 10 seconds. Repeat it (press Flush Rig to stop it, then again to start it) until water comes out of both spouts and there is no air left in the tubes. Let each run finish its 10 seconds before stopping it: stopping it early can leave the valves open. Flush Rig has to be stopped before you launch a session.

### For each mouse

1. **Weigh** the mouse and record the weight.
2. **Load its parameters, before head-fixing.** Choose the experimenter and the mouse and press **Load params** (INSTRUCTIONS steps 2–4). If Load refuses, the problem is in the spreadsheet or in `Presets.csv` and has to be fixed in the repository; better to find that out before the mouse is in the rig.
3. **Head-fix the mouse and position the spouts.** It should be the same distance from both spouts and able to reach both. Do this before launching: moving the spouts during a session changes the task partway through the data.
4. **Launch** (INSTRUCTIONS step 6) and watch the first trials. Bonsai shows the camera, each trial's summary row as it is saved (including `Trial_Number`), and graphs of performance over the last trials and for each range of sound levels.
5. **Stop at 30–40 minutes or 600 trials, whichever comes first.** Nothing stops the session by itself. A mouse that doesn't lick at all still stays on the rig for 10–15 minutes, to get used to it. End the session and kill Bonsai (INSTRUCTIONS step 7).
6. **Put the mouse back** in its cage. Top up its water if it drank little, and record the day in the restriction sheet.
7. **Clean the rig** with 70% ethanol, especially droppings and urine. Mice urinate more in their first sessions, when they are stressed.

### At the end of the day

Press **Push Data** on every rig you used (INSTRUCTIONS step 8). It only adds files to the server and never deletes anything from the rig.

## When a rig's disk is full

Clear old sessions from the rig's `Data\` folder yourself. First check they are on the server: press Push Data, and if it copies no new files and lists no files with a different size, everything on the rig is already there.

## Calibration

To be written from the water and sound calibration protocols. New valve times and speaker values go into the rig's row of `Params/Rigs.csv`, as a change in the repository, like adding a mouse.

## Moving through training

Each step is a preset in `Params/Presets.csv`, and its name is saved as the Session Type, so the data record which step every session was. To move a mouse on, change its Preset cell in the spreadsheet, in the repository (see [Adding a mouse](#adding-a-mouse)). [Mouse_Room_Params_Example.xlsx](Mouse_Room_Params_Example.xlsx) shows every value at each step.

The criteria for moving from one step to the next are still to be added.

| Step | What changes | Typical length |
|---|---|---|
| `habituation` | Water every 10 s from either spout; no lick needed. ITI 10 s (2 s from the next step on). | about 1 day |
| `lick-to-release` | Every lick is rewarded, on the side licked. | 1–2 days |
| `three-and-three` | The mouse has to lick each side 3 times in a row. | 3–5 days |
| `regular-antibias-200` to `regular-antibias-025` | The full task, with sound levels drawn evenly across the range. Anti-bias on, its slope lowered step by step: 2.00, 1.75, 1.50, 1.00, 0.50, 0.25. Sound 300 ms, then 150 ms from `regular-antibias-100` on. | |
| `regular` | Anti-bias off. | |
| `hard-A`, `hard-B` | Asymmetric distribution: `hard-A` is Asym_Right (low sounds drawn close to the boundary), `hard-B` is Asym_Left (high sounds drawn close to it). | |

The experiment presets build on `regular`, `hard-A` and `hard-B`:

| Presets | Session |
|---|---|
| `opto-regular`, `opto-hard-A`, `opto-hard-B` | Zapit on 30% of trials: PPC, bilateral, light from Sound to Feedback |
| `masking-regular`, `masking-hard-A`, `masking-hard-B` | the same panel settings as opto; masking comes from the Zapit configuration you load |
| `washout-regular` | as `regular`, no light |
| `control-alm-bi`, `control-alm-uni` | Zapit on 30% of trials at ALM: both sides, or the right side only |
