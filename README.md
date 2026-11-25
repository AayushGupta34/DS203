# DS203 Project: Classifying countries by HDI class

Course project for DS203 at IIT Bombay (classification problem, 2026).

Team: Aayush Gupta (25B0064), Maanas Katwala (25B0336), Avadhoot Nikam (25B2108)

We split 193 countries into three classes using their 2023 HDI score, then tried to predict the class from nine World Bank indicators plus the country's region and its dominant economic sector. HDI and its components (life expectancy, schooling, income per person) are not used as inputs, since they would give the answer away.

| y | Class | HDI 2023 | Countries |
|---|-------|----------|-----------|
| 0 | Low | below 0.550 | 26 |
| 1 | Medium/High | 0.550 to 0.800 | 93 |
| 2 | Very High | 0.800 and above | 74 |

These three classes were defined for the project. They are not an official UNDP grouping.

## Data

- HDI: UNDP Human Development Report 2025 ([download page](https://hdr.undp.org/data-center/documentation-and-downloads)), file `HDR25_Composite_indices_complete_time_series.csv`.
- Indicators: World Bank World Development Indicators ([DataBank](https://databank.worldbank.org/source/world-development-indicators)), years 2021 to 2023, file `WDI_Data_2021_2023.csv`. The nine indicators are GDP growth, inflation, unemployment, infant mortality, access to electricity, internet users, agriculture share of GDP, industry share of GDP and trade (% of GDP).
- Region: World Bank country API. The notebooks fetch this directly, so there is no file for it.

For each indicator we take the 2023 value. If 2023 is missing we use 2022, then 2021. The year that was actually used is saved next to every value. `DS203_data_dictionary.csv` lists all columns with their source and this year rule.

## Running the code

You need Python 3 with these packages:

```
pip install numpy pandas matplotlib seaborn scipy scikit-learn requests
```

Keep the notebooks and CSV files in the same folder, because all paths are relative. The first three notebooks call the World Bank API for region data, so they need an internet connection.

Run the notebooks in this order. Each one saves a CSV that the next one reads. All of these CSVs are already in the repo, so any notebook can also be opened and run on its own.

1. `DS203_dataset_setup.ipynb` merges the HDI and WDI data on ISO3 country code, drops 24 territories that have no HDI value, creates the target `y` and the `DominantSector` column, and saves `DS203_final_dataset.csv` and `DS203_data_dictionary.csv`.
2. `DS203_EDA.ipynb` looks at class balance, missing values, distributions, outliers, how each indicator differs between classes, and correlations.
3. `DS203_preprocessing.ipynb` applies log(x+1) to the skewed columns, makes a stratified 60/20/20 train/validation/test split (seed 42), compares imputation options with cross-validation, and runs PCA. It saves `DS203_model_data.csv`, which has a `split` column so the training notebook uses exactly the same rows.
4. `DS203_training.ipynb` has the majority-class baseline, tunes logistic regression, a decision tree and a random forest with 5-fold `GridSearchCV` (macro F1), evaluates them once on the test set, and then looks at coefficients, permutation importance and the misclassified countries.

Imputation, scaling and one-hot encoding are all inside a scikit-learn `Pipeline`, so they are fitted on training data only.

## Results

Test set, 39 countries:

| Model | Accuracy | Balanced accuracy | Macro F1 |
|-------|----------|-------------------|----------|
| Majority baseline | 0.487 | 0.333 | 0.218 |
| Logistic regression | 0.795 | 0.806 | 0.769 |
| Decision tree | 0.795 | 0.801 | 0.781 |
| Random forest | 0.795 | 0.752 | 0.765 |

The three models end up very close on the test set, and all of them are well above the baseline (which always predicts Medium/High). The random forest had the best cross-validation score (0.887 macro F1). Infant mortality and access to electricity were the most useful features. Most of the wrong predictions are countries whose HDI sits near the 0.550 or 0.800 cut-off.

The test set has only 5 Low countries, so one mistake there changes the scores a lot. The report goes through this and the other limitations in more detail.

## Other files

| File | What it is |
|------|------------|
| `HDR25_Composite_indices_complete_time_series.csv` | Raw UNDP HDI data |
| `WDI_Data_2021_2023.csv` | Raw World Bank indicator data |
| `DS203_final_dataset.csv` | Merged dataset, one row per country |
| `DS203_data_dictionary.csv` | Column descriptions, sources and year rule |
| `DS203_model_data.csv` | Model-ready data with the train/validation/test split |
| `DS203_2026_project-classification.pdf` | Problem statement |
| `DS203_2026_project_instructions (2).pdf` | Course project instructions |

`hdr-data.xlsx` and `470b9aeb-..._Series - Metadata.csv` came with the original downloads. The notebooks don't read them.
