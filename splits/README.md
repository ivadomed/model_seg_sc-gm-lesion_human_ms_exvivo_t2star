# Cross-validation splits (subject-level, recovered & verified)

The paper's models were trained with a **subject-level 4-fold** CV split (the spinal cord is the unit).
Two splits exist because the 2D and 3D datasets were built separately:

- `subject_split_2D.json` — the **2D** models' split (sorted subjects, no shuffle; from the historical
  `create_subject_split.py`). Verified: re-predicting held-out validation with it reproduces the paper's
  2D cross-fold numbers. **Using any other split leaks** (the current `Dataset021` split differs!).
- `subject_split_3D.json` — the **3D** models' split (shuffled). Recovered from the stored held-out
  `predictions_3d`; verified to reproduce the paper's 3D lesion numbers.

Materialize an nnU-Net case-level `splits_final.json` for a dataset from either file with
`python -m helpers.make_splits --dataset-dir <raw dataset> --canonical splits/subject_split_<dim>.json --inject`
(uses that dataset's `inference_manifest.json`/`manifest.json`).

> Future work should unify 2D and 3D onto one split; kept as-is here to reproduce the published models.
