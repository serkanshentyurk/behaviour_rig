# Handover

Where the experiments stand when the person running them leaves, and what is still to decide. Fill in everything marked **to confirm**, and keep this page current until the next experimenter has taken over.

It doesn't replace the guides: [TRAINING.md](TRAINING.md) for the daily routine, [INSTRUCTIONS.md](INSTRUCTIONS.md) for the control panel, [HARDWARE.md](HARDWARE.md) for the rigs, [ZAPIT.md](ZAPIT.md) for opto, and `START_HERE.md` in the `sound_categorisation` repository for the data and the analysis.

## The mice

The presets below are what `Params/Mouse_Room_Params.xlsx` holds in the repository on 10 November 2026. The other columns are **to confirm**.

| Mouse | Rig | Preset now  | Last usable session | Next planned session | Notes |
| ----- | --- | ----------- | ------------------- | -------------------- | ----- |
| SS24  |     | habituation |                     |                      |       |
| SS25  |     | habituation |                     |                      |       |
| SS26  |     | habituation |                     |                      |       |
| SS27  |     | habituation |                     |                      |       |
| SS28  |     | habituation |                     |                      |       |
| SS29  |     | habituation |                     |                      |       |
| SS30  |     | habituation |                     |                      |       |
| SS31  |     | habituation |                     |                      |       |
| SS32  |     | habituation |                     |                      |       |
| SS33  |     | habituation |                     |                      |       |
| SS34  |     | habituation |                     |                      |       |
| SS35  |     | habituation |                     |                      |       |

## What has been agreed

- **Order of the Hard distributions**: Discuss with Athena. The previous cohorts showed no measurable adaptation with daily alternation and at most a partial one with blocks (`START_HERE.md`, Status).
- **Sessions per phase or block:** Discuss with Athena.
- **Exclusion criteria** for sessions and mice: **to confirm**. The analysis applies its own checks (the data caveats and the bias check in `START_HERE.md`); anything beyond those goes here.

## Where things are

- **Rig code:** this repository. It will move to the lab's GitHub (**to confirm** the address); then repoint every rig ([HARDWARE.md](HARDWARE.md#when-the-repository-moves)).
- **Analysis code:** `sound_categorisation` and `behav_utils`, moving to the lab's GitHub as well.
- **Data:** `/ceph/akrami/Serkan/Head_Fixed_Behavior/Data`, with the sessions in `Raw` and the analysis in `Processed`.
- **Zapit configurations:** `Params/zapit/` ([ZAPIT.md](ZAPIT.md#configurations)).

## Who owns what afterwards

- **The rigs and the room:** Until now Quentin Pajot-Moric and Serkan Shentyurk; see the README, [Contacts](README.md#contacts). Now you?
- **The analysis:** Until now Quentin Pajot-Moric and Serkan Shentyurk -- see [Sound Categorisation](https://github.com/serkanshentyurk/sound_categorisation) repo.
- **Access, and the cluster:** as in the README, [Contacts](README.md#contacts).
