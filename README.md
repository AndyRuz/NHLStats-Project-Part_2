# NHL Data Analysis (2004-2018)
### CS3120 Final Project — Part 2
### Andy Ruzicka


This repository contains the Jupyter notebook for my CS3120 final project,
which uses NHL skater data to investigate three machine-learning hypotheses
about player performance.

## Files
**`Original data pulled from Kaggle (https://www.kaggle.com/datasets/xavya77/nhl04to18) [Author: Xavya77]`**

- **`nhl-final-project-andy-ruzicka.ipynb`** — the main notebook containing EDA, model
  fitting, and conclusions.
- **`nhl_data.csv`** — the cleaned source dataset (NHL 2014-2015 thru 2017-2018 seasons)

## Research Questions

1. **Can a player's total points (Goals + Assists) be predicted from peripheral
   statistics** like time on ice, shots on goal, age, and position?
2. **Can we classify whether a player is a forward or a defenseman** using
   only their statistical profile (without seeing the position label)?
3. **Does age have a measurable relationship with offensive production**,
   and is there a "golden age" curve?

## Methods

- **Exploratory Data Analysis** — distributions, correlations, and
  position-stratified plots of all key statistics (5 figures across 5
  EDA subsections).
- **Modeling** — Linear Regression, Random Forest, and Gradient Boosting
  for Q1; Logistic Regression and Random Forest for Q2; Gradient Boosting
  with Permutation Importance + PDP + ICE plots for Q3.
- **Evaluation** — train/test splits, 5-fold cross-validation, and **R²**
  reported alongside RMSE / accuracy / AUC for each model.

## Headline Results

| Question | Best Model | Metric | Value |
|----------|------------|--------|-------|
| Q1: Predict PTS | Gradient Boosting | CV R² | 0.89 |
| Q2: Classify F vs D | Logistic Regression | Test AUC | 0.98 |
| Q3: Age effect on PPG | Gradient Boosting | CV R² | 0.75 |

For the full discussion of findings, see the "Conclusions" section in the notebook.

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook nhl-final-project-andy-ruzicka.ipynb
```

## AI Tools Used

Claude was used for this project for exploration, modeling design, and code generation.
