# Dataset Information

## Semiconductor Manufacturing Process Dataset (SECOM)

The dataset used in this project is the **Semiconductor Manufacturing Process (SECOM)** dataset.

### Specifications:
- **File Name:** `signal-data.csv`
- **Total Observations:** 1,567 rows
- **Total Attributes:** 592 columns (590 numerical sensor measurements, 1 `Time` timestamp, 1 `Pass/Fail` target)
- **Target Variable:** `Pass/Fail` (-1 = Pass / Normal, 1 = Fail / Defective)
- **Class Imbalance:** 1,463 Pass (93.36%) vs 104 Fail (6.64%)

### Usage Instructions:
1. Obtain `signal-data.csv` (UCI SECOM Dataset).
2. Place the file in the project root directory or in `data/signal-data.csv`.
3. Run the Jupyter Notebook `Corizo_Capstone_1_Minor_DataScience.ipynb`.

> **Note:** The dataset contains raw sensor measurements and is excluded from git tracking to keep the repository size optimal.
