# Dataset Download Instructions

## Why the Raw Dataset Is Not Included

The raw dataset used for this capstone project is not included in this GitHub repository because the file is too large for regular GitHub storage.

The project uses the following raw dataset file:

```text
2020-Apr.csv
```

This file should be downloaded separately from Kaggle and placed in the local project folder before running the notebooks.

---

## Dataset Source

Dataset name:

```text
eCommerce behavior data from multi category store
```

Dataset author:

```text
Michael Kechinov
```

Dataset platform:

```text
Kaggle
```

Dataset URL:

```text
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
```

---

## Dataset Description

The dataset contains user behavior events from a large multi-category eCommerce store. Each row represents a customer interaction event related to a product and user session.

Common event types include:

- product views
- cart additions
- purchases

The available dataset files cover customer behavior from October 2019 to April 2020.

This project specifically uses the April 2020 file:

```text
2020-Apr.csv
```

---

## Where to Place the Dataset

After downloading the dataset from Kaggle, place the raw file in this folder:

```text
project_folder/data/raw/2020-Apr.csv
```

The expected local project structure is:

```text
project_folder/
│
├── data/
│   ├── raw/
│   │   └── 2020-Apr.csv
│   │
│   └── processed/
│
├── notebooks/
├── models/
├── reports/
├── README.md
└── requirements.txt
```

---

## How to Download from Kaggle

### Option 1: Manual Download

1. Go to the Kaggle dataset page:

```text
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
```

2. Sign in to Kaggle.
3. Click **Download**.
4. Extract the downloaded files if they are zipped.
5. Copy `2020-Apr.csv` into:

```text
project_folder/data/raw/
```

---

### Option 2: Kaggle API Download

If the Kaggle API is installed and configured, the dataset can be downloaded using:

```bash
kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store
```

Then unzip the downloaded file and move `2020-Apr.csv` into:

```text
project_folder/data/raw/
```

Example:

```bash
mkdir -p project_folder/data/raw
unzip ecommerce-behavior-data-from-multi-category-store.zip -d project_folder/data/raw
```

For Windows PowerShell, you can manually extract the ZIP file or use:

```powershell
Expand-Archive -Path ecommerce-behavior-data-from-multi-category-store.zip -DestinationPath project_folder/data/raw
```

---

## Important Note

The raw dataset and large intermediate processed files are intentionally excluded from GitHub using `.gitignore`.

This keeps the repository lightweight and avoids uploading very large files. The notebooks can regenerate the processed outputs once the raw `2020-Apr.csv` file is placed in the correct folder.
