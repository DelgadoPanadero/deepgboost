# Reference: Kaggle search queries for DeepGBoost research

## Competitions where boosting dominates (to understand baseline)

```
site:kaggle.com discussion "1st place" LightGBM tabular 2022 OR 2023 OR 2024
site:kaggle.com discussion "1st place" XGBoost tabular structured data
site:kaggle.com competition tabular "gradient boosting" wins solution
```

## Competitions where neural networks beat boosting (DeepGBoost opportunity)

```
site:kaggle.com discussion "neural network" beats "gradient boosting" tabular
site:kaggle.com discussion "deep learning" tabular "better than XGBoost"
site:kaggle.com competition "neural" 1st place tabular structured
```

## Ensemble and hybrid approaches

```
site:kaggle.com discussion "ensemble" "gradient boosting" "neural" tabular solution
site:kaggle.com "stacking" "LightGBM" "neural network" tabular winning
site:kaggle.com "blend" XGBoost neural tabular 1st place
```

## Specific high-profile tabular competitions to check

| Competition | Slug | What to look for |
|---|---|---|
| Tabular Playground Series | tabular-playground-series-* | Monthly comps, well-documented solutions |
| Santander Customer Satisfaction | santander-customer-satisfaction | Classic GBM vs NN benchmark |
| Porto Seguro | porto-seguro-safe-driver-prediction | Feature engineering + boosting |
| IEEE Fraud Detection | ieee-fraud-detection | XGBoost dominance, lessons learned |
| House Prices | house-prices-advanced | Regression, regularization tricks |
| Higgs Boson | higgs-boson-machine-learning-challenge | Early DNN vs boosting comparison |
| AmEx Default | amex-default-prediction | Recent, high-prize tabular comp |

## Meta-analysis queries

```
# What fraction of Kaggle competitions are won by boosting?
site:kaggle.com "won with" OR "1st place" "gradient boosting" OR "LightGBM" OR "XGBoost" 2023

# Survey-style discussion posts
site:kaggle.com "tabular data" "which model" "works best" discussion
```
