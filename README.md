# D-BIAT analysis code

Code for the article *Implicit Death Associations in Young Adults: Evaluating the Death-Brief IAT for Differentiating Mental Health Status* (Kadiri et al., Frontiers in Psychiatry).

Seven Jupyter notebooks reproduce the analyses of the 135 analyzed participants reported in the article and its Supplementary Material.

## Data

The data are **not publicly available** because they contain sensitive information on depression and suicidal ideation. They are available upon reasonable request to shri@usc.edu.

The analyses use one file, `DBIAT_data_N135.csv`, with all 120 D-BIAT trials of each of the 135 participants (one row per trial). Its columns are:

| Column | Meaning |
|---|---|
| `participant` | Study code |
| `group` | Control, Depressed, or Suicidal |
| `task_version` | `browser` (jsPsych) or `MATLAB` (Psychtoolbox-3) |
| `block_number`, `block_type` | Block 1–6; `Life:Me` or `Death:Me` |
| `trial_in_block` | Trial 1–20 |
| `word`, `word_category` | Word presented; Me, Other, Life, or Death |
| `target_key` | Key shared by the two target categories in the block |
| `correct` | 1 if the first response was correct, 0 otherwise |
| `rt_ms` | Reaction time (ms) from word onset |
| `stimulus_onset_s`, `time_elapsed_ms` | Time stamps (MATLAB and browser versions), used only for the task-timing description |

## How to run

1. Install Python 3.11 and the packages: `pip install -r requirements.txt`
2. Create a folder `data/` and put `DBIAT_data_N135.csv` in it.
3. Run all notebooks: `bash run_all.sh` (about 1 minute), or open them in Jupyter and run them in order.

Results are written to `outputs/` (`tables/`, `figures/`, `derived/`).

## Notebooks

| Notebook | Content |
|---|---|
| `01_Preprocessing` | Removes the last trial of each block and trials outside 400–2000 ms; checks the participant-level criteria |
| `02_Scores` | Participant measures and D-scores; task version by group |
| `03_Within_Group` | Life:Me vs. Death:Me within groups (Figure 2, Table S5) |
| `04_Between_Group` | Group comparisons (Figures 3–5, Table S7) |
| `05_Classification` | AUCs, predictive values, ROC curves (Table 3, Tables S4 and S8, Figure S2) |
| `06_Assumptions` | Normality and variance checks (Table S6) |
| `07_Task_Description` | Task and timing values (Tables S1 and S2) |

Every analysis setting (pre-processing rules, random seeds, numbers of resamples) is in `settings.yaml`, and the shared functions are in `dbiat_analysis.py`. Because the seeds are fixed, the resampling results are reproduced exactly.

## Contact

Sudarsana Reddy Kadiri, skadiri@usc.edu
