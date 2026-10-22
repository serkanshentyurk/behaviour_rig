# Training a mouse

What an experimenter does with a mouse, from surgery to the experiments, once the rigs are set up and working. For the control panel itself, see [INSTRUCTIONS.md](INSTRUCTIONS.md); this page points to its steps instead of repeating them. The rigs are described in [HARDWARE.md](HARDWARE.md), Zapit in [ZAPIT.md](ZAPIT.md) and the surgery in [SURGERY.md](SURGERY.md).

## Before you start

- You need the SWC inductions and the PIL course and assessments before working with the mice. They are not covered here: ask the PI or the lab manager, or see the lab manual.
- Animal welfare follows the project licence (PPL) and the facility's rules. This page only covers what is specific to this task.
- You get access to the room with the induction. Cages can be moved from the holding room to the rig room.
- While you use the room, you look after it: cleaning, taking the rubbish out, keeping gloves stocked.
- Access to the repository, the rig PCs, the server and the restriction sheet: see the README, [Access you need](README.md#access-you-need).

## The mice

We use VGAT-ChR2-EYFP mice, as young as possible. Sex and housing (single or group) don't matter.

Each cage has a cage ID and each mouse an animal ID. These are official and you don't change them. Day to day, and on the rigs, each mouse also has a nickname: your initials and the next number, e.g. SS14. In PyRAT it is written with the ear mark, e.g. `SS14 (L)`; the ear marks are L, R, LR, LL and RR.

## Records

Every mouse on water restriction goes in **Dammy Mouse Restriction.xlsx**, on SharePoint under SWC PIL › restriction sheets › Akrami Lab.

- Once per mouse: cage ID, mouse ID, nickname (e.g. SS14) and start weight.
- Every day the mouse is on restriction: its weight, and whether it was trained or given water.
- When it comes off restriction: `break` or `end` (see [Water restriction](#water-restriction)).

Any other notes you keep, and where, are up to you.

## Adding a mouse

Do this in the repository on your own computer, before the mouse's first session. Not on a rig: each rig resets to the repository's main branch whenever its launcher starts, so an edit made on a rig is lost.

1. In `Params/Mouse_Room_Params.xlsx` (sheet `Params`), copy an existing mouse's row and change **Subject** to the new nickname.
2. Set **Preset** to `habituation`. Leave Sound_Contingency as it is: all mice so far use `Low_Left_High_Right`.
3. From the repository root, run `python -m pytest tests`. The tests check that every mouse loads.
4. Commit and push. Each rig picks the change up the next time its launcher starts.

Push Data only copies mice whose nickname starts with the experimenter's initials, so a mouse named otherwise never reaches the server.

## Surgery

The mice get a head-plate. Mice that will run Zapit sessions also get marks on the skull during the same surgery. See [SURGERY.md](SURGERY.md).

## From surgery to the first session

| Days after surgery | What to do |
|---|---|
| 1–3 | **Handling.** Put your hand in the cage and let the mouse sniff it and walk on it, then start picking it up and handling it. |
| 4 | **Water restriction starts.** Weigh the mouse and record its start weight. Keep handling it, and give it water from your hand, as much as it will drink. Put it on the rig without head-fixing it, let it explore, and give it water there. |
| 5 | **Head-fixing.** As on day 4, but now head-fix the mouse and give it water while it is head-fixed (on a piece of paper, for example), again until it stops drinking. |
| 6 | **First training session**, with `habituation`. |

**Head-fixing.** Weigh the mouse and note the weight. Put it on the rig, hold the head-plate gently and slide it into the groove of the holder. Turn the metal clamp over the groove and tighten its screw by hand until it is firm.

## Water restriction

- Weigh every mouse every day it is on restriction, before any session, and record the weight. Check its general condition at the same time.
- Its weight must not go below 80% of the start weight. If it does, give it more water until it is back above 80%.
- On days without training, weekends included, give each mouse 1 minute with the water bottle in its cage, and record it as usual.
- After a session: if the mouse got about 300 correct trials, it has had enough water. Otherwise top it up a little from the bottle; there is no fixed amount.
- When a mouse comes off restriction, write `break` or `end` in the restriction sheet. At the end of the experiment, take it off restriction and write `end`.
- How long a mouse may stay on restriction, and when it needs a break, are set by the PPL.

## Every week

- Calibrate water and sound on each rig (see [HARDWARE.md](HARDWARE.md#calibration)).

## Every day

### In the morning, on each rig

1. **Fill the water syringe** to the top (20 ml). Keep it above 15 ml during the day: the rigs are calibrated with a full syringe, and the reward gets smaller as the level drops. Leave it full at the end of the day.
2. **Start the panel** ([INSTRUCTIONS.md](INSTRUCTIONS.md) step 1).
3. **Flush.** On the Setup tab press **Flush Rig**. Each time it starts, it opens both valves for about 10 seconds. Repeat until water comes out of both spouts and there is no air left in the tubes.
   - Let each run finish its 10 seconds before stopping it; stopping it early can leave the valves open.
   - Flush Rig has to be stopped before you launch a session.
   - If no water comes out on a Harp rig (31101, 31102), see [The Harp Behaviour app](HARDWARE.md#the-harp-behaviour-app).
4. **Check the lick sensors.** Touch each spout with a finger; its touch-sensor light should come on. If you want to check the camera, use the **Camera** button on the Setup tab, and stop it again before launching.

### For each mouse

1. **Weigh** the mouse and record the weight.
2. **Load its parameters, before head-fixing.** Choose the experimenter and the mouse and press **Load params** (INSTRUCTIONS steps 2–4). If Load refuses, the problem is in the spreadsheet or in `Presets.csv` and has to be fixed in the repository; better to find that out before the mouse is in the rig.
3. **Head-fix the mouse and position the spouts.** Move the spouts with the mover and its fine-adjustment knobs. They should sit slightly below the mouth, the same distance from both sides, so the mouse can reach and lick both. The touch-sensor lights show which side it licks. Do this before launching: moving the spouts during a session changes the task partway through the data.
4. **Launch** (INSTRUCTIONS step 6) and watch the first trials. Bonsai shows the camera, each trial's summary row as it is saved (including `Trial_Number`), and graphs of performance over the last trials and for each range of sound levels.
   - In the first sessions (`habituation`, `lick-to-release`, `three-and-three`, perhaps the first full-task session) you can reward the mouse by touching the correct spout yourself. The rig counts the touch as a lick.
5. **Stop at 30–40 minutes or 600 trials, whichever comes first.** Nothing stops the session by itself. A mouse that doesn't lick at all still stays on the rig for 10–15 minutes, to get used to it. End the session and kill Bonsai (INSTRUCTIONS step 7).
6. **Put the mouse back** in its cage, top up its water if needed (see [Water restriction](#water-restriction)), and record the day in the restriction sheet.
7. **Clean** the head-plate holder with 70% ethanol between mice, and clean droppings and urine off the rig. There is more of that in the first sessions, when mice are stressed.

It is better to keep each mouse on the same rig, but switching rigs is not a problem.

### At the end of the day

Press **Push Data** on every rig you used (INSTRUCTIONS step 8). It only adds files to the server and never deletes anything from the rig.

## When a rig's disk is full

Clear old sessions from the rig's `Data\` folder yourself. First check they are on the server: press Push Data, and if it copies no new files and lists no files with a different size, everything on the rig is already there.

## Moving through training

Each step is a preset in `Params/Presets.csv`, and its name is saved as the Session Type, so the data record which step every session was. To move a mouse on, change its Preset cell in the spreadsheet, in the repository (as in [Adding a mouse](#adding-a-mouse)). [Mouse_Room_Params_Example.xlsx](Mouse_Room_Params_Example.xlsx) shows every value at each step.

Judge progress from Bonsai's windows during the session: the text window shows the trial count, and the bar graph performance in each of the 8 stimulus bins between −1 and +1. Or analyse the sessions afterwards with [behav_utils](https://github.com/serkanshentyurk/behav_utils). There is no single right way.

| Step | What changes | Typically | Move on after |
|---|---|---|---|
| `habituation` | Water every 10 s from either spout; no lick needed. ITI 10 s (2 s from the next step on). | 1–2 days | at least 300 trials |
| `lick-to-release` | Every lick is rewarded, on the side licked. | 1–2 days | at least 300 trials |
| `three-and-three` | The mouse has to lick each side 3 times in a row. | 1–2 days | at least 300 trials |
| `regular-antibias-200` to `regular-antibias-025` | The full task, with sound levels drawn evenly across the range. Anti-bias on, its slope lowered step by step: 2.00, 1.75, 1.50, 1.00, 0.50, 0.25. Sound 300 ms, then 150 ms from `regular-antibias-100` on. | | see below |
| `regular` | Anti-bias off. | | see below |

From `regular-antibias-200` on, move to the next step when, for two sessions in a row:
- the mouse gets at least 200–250 correct trials;
- it is correct on about 70% of trials, and on more than 60% in the two inner bins (4 and 5);
- it shows little side bias: the bins on the two sides look alike.

If a side bias grows, go back one step (more anti-bias). Don't go back as far as `habituation`, `lick-to-release` or `three-and-three`. Whether and when to exclude a mouse is up to the experimenter and the project.

## Experiments

After `regular`, the usual order is:

1. `masking-regular`
2. `opto-regular`
3. `washout-regular`, after the opto sessions, to wash out any lasting effect of the light. It is not strictly needed.
4. The hard sessions: `hard-A`, `hard-B`, and their `masking-` and `opto-` versions. Every mouse does both A and B; the order and the number of sessions are up to the experimenter.

`control-alm-bi` can be run at any point once the mouse performs well. Run `control-alm-uni` last: stimulating one side can leave the mouse with a bias.

| Presets | Session | Rig |
|---|---|---|
| `masking-regular`, `masking-hard-A`, `masking-hard-B` | the same panel settings as opto; the masking comes from the Zapit configuration | 31108 |
| `opto-regular`, `opto-hard-A`, `opto-hard-B` | Zapit on 30% of trials: PPC, bilateral, light from Sound to Feedback | 31108 |
| `washout-regular`, `hard-A`, `hard-B` | no light | any |
| `control-alm-bi`, `control-alm-uni` | Zapit on 30% of trials at ALM: both sides, or the right side only | 31108 |

Which Zapit configuration goes with each preset is in [ZAPIT.md](ZAPIT.md#configurations). The opto and masking presets all use Zapit, so for now they only run on 31108. 31101 and 31102 have fibre optogenetics, but there are no fibre presets yet.

## Changing or adding presets

The presets are in `Params/Presets.csv`, one row per preset. Change them in the repository, like the spreadsheet.

- **To change a preset,** edit its row and run the tests. Check `git diff Params/Presets.csv` before pushing: the change applies to every mouse on that preset from its next session.
- **To add a preset:**
  1. Add its name to `SESSION_TYPES` in `GUI/params_io.py`. A preset's name is also its Session Type, so it has to be a Session Type option first.
  2. Add the same name to the Session Type options in [PARAMETERS.md](PARAMETERS.md#session-tab). The tests fail until it's there, and say exactly what to write.
  3. Add a row for it to `Presets.csv`. An empty cell leaves that parameter to each mouse's own row; `NaN` sets it to empty.
  4. Run the tests, then commit and push.
- The criteria for moving through a new preset are up to the experimenter.
- More on how presets work: [PARAMETERS.md](PARAMETERS.md#presets).
