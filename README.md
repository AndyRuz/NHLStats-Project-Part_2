# NHL Player Performance Analysis (2004-2018)
### CS3120 Final Project — Part 2

This repository contains the Jupyter notebook for my CS3120 final project,
which uses NHL skater data to investigate three machine-learning questions
about player performance.

## Files

- **`NHL_Final_Project.ipynb`** — the main notebook containing EDA, model
  fitting, and conclusions. View it directly in GitHub or open it in
  Jupyter / Colab.
- **`nhl_data.csv`** — the cleaned source dataset (NHL skater season-level
  statistics, 2004-05 through 2017-18), originally from
  [Kaggle](https://www.kaggle.com/datasets/xavya77/nhl04to18).

## Research Questions

1. **Can a player's total points (PTS) be predicted from peripheral
   statistics** like time on ice, shots on goal, age, and position?
2. **Can we classify whether a player is a forward or a defenseman** using
   only their statistical profile (without seeing the position label)?
3. **Does age have a measurable relationship with offensive production**,
   and is there a "peak age" curve?

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

For the full discussion of findings — including the surprising result that
Q3's apparent "peak age" effect is largely an opportunity / selection
artifact — see the conclusions section in the notebook.

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook NHL_Final_Project.ipynb
```

## AI Tools Used

Claude (Anthropic) was used throughout this project for exploration,
modeling design, code generation, and writing.
