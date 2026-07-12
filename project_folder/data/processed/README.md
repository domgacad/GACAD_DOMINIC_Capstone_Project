# Processed Data Files

## Why Some Processed Files Are Not Included

Some processed data files generated during this project are not included in the GitHub repository because they exceed GitHub's regular file size limit of 100 MB per file.

The following processed files were excluded from the repository:

```text
project_folder/data/processed/event_sample_cleaned_for_eda_feature_engineering.csv
project_folder/data/processed/event_sample_session_preserved_approx_1m.csv
```

These files are intermediate processed datasets created during the data cleaning, preprocessing, exploratory data analysis, and feature engineering workflow.

---

## External Download Link

The large processed data files can be downloaded from this Google Drive folder:

```text
https://drive.google.com/drive/folders/1dffNxIu9WTaL7RZNeUjzMF9rHY-z1Hg9
```

---

## Expected Local File Location

After downloading the files, place them in the following folder:

```text
project_folder/data/processed/
```

The expected local structure is:

```text
project_folder/
└── data/
    └── processed/
        ├── event_sample_cleaned_for_eda_feature_engineering.csv
        ├── event_sample_session_preserved_approx_1m.csv
        └── other processed summary outputs
```

---

## Notes for Reproducibility

These large files are not required to view the notebooks, reports, model evaluation summaries, model artifacts, and presentation materials included in the GitHub repository.

However, they may be useful if the full preprocessing, exploratory analysis, and feature engineering workflow needs to be reviewed or reproduced from the intermediate processed-data stage.

The smaller processed summary files, evaluation outputs, feature importance files, model metadata, and reports are included in the repository where possible.
