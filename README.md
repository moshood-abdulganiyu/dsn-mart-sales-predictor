# DSN Mart Sales Predictor

Regression model built for the **DSN Bootcamp Qualification Hackathon 2026 — ML Track**, predicting `total_sales` for DSN Mart (product × store rows), scored by **RMSE** (lower is better).

## Task

- `train.csv`: 6,818 rows, 13 columns
- `test.csv`: 1,705 rows, 12 columns (no `total_sales`)
- Submission format: `id`, `total_sales` — must match `sample_submission.csv` row order exactly

## Repo structure

```
dsn_ai_bootcamp/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── DSN_Mart_Sales_Predictor.ipynb   # full pipeline, 9 sections
├── models/
│   ├── catboost_final_model.joblib
│   └── catboost_feature_config.joblib
├── submission.csv
├── main.py
├── pyproject.toml
└── uv.lock
```

## Data quality decisions

| Issue | Decision | Why |
|---|---|---|
| `product_weight_kg` ~18% missing | Imputed via `product_code` group lookup (train+test combined), category-median fallback | Weight is a fixed product property — safe, no leakage |
| `store_size` missing (100% for 3/10 stores) | Kept as its own `"Missing"` category, **not imputed** | EDA confirmed missing rows average ~21% lower sales (2310→1828 mean) — real signal, not noise |
| `product_category` 48 raw → 16 real values | Normalized via `.str.lower().str.strip()` | Case/whitespace variants were fragmenting category signal |
| `fat_content` | Left as-is | Confirmed identical clean value sets (`Low Fat`/`Regular`) on train and test |
| `shelf_visibility` exact zeros (~6%) | Left as raw, no imputation, no flag | EDA + CV testing showed no meaningful sales difference between zero and non-zero rows |
| `product_code` (1,555 unique) | Used natively as a CatBoost categorical feature | Manual target/frequency encoding for tree models actively hurt performance |
| `id` | Dropped from features | Row identifier only |

## Model comparison

| Model | Features | CV RMSE | Std |
|---|---|---|---|
| Mean baseline | none | 1697.72 | 26.17 |
| Ridge | one-hot + numeric + freq-encoded `product_code` | 1125.83 | 26.06 |
| Random Forest | target-encoded `product_code`/`store_code` + one-hot | 1214.14 | 47.70 |
| CatBoost (manual target encoding) | target-encoded + one-hot | 1210.41 | 50.50 |
| CatBoost (manual encoding, log1p target) | same, log1p target | 1229.57 | 44.95 |
| **CatBoost (native categoricals)** | raw categorical columns, raw target | **1092.07** | **18.61** |
| CatBoost (native cat, log1p target) | same, log1p target | 1120.83 | 25.76 |
| CatBoost (native cat, imputed visibility) | raw cat + imputed `shelf_visibility` | 1093.71 | 19.82 |
| CatBoost (native cat, visibility + is_zero flag) | raw cat + raw visibility + flag | 1092.93 | 20.67 |
| CatBoost tuning: iters=1200, lr=0.03 | — | 1088.69 | 21.11 |
| CatBoost tuning: depth=8 | — | 1102.61 | 24.62 (overfits) |
| CatBoost tuning: l2_leaf_reg=8 | — | 1087.90 | 21.74 |
| CatBoost tuning: iters=600, lr=0.08, depth=5 | — | 1090.37 | 20.75 |

**Key finding:** switching CatBoost from manually target-encoded features to **native categorical handling** was the single biggest win after the initial Ridge baseline (1210 → 1092) — bigger than any hyperparameter tuning. All tuning variants landed within noise (~1088–1102), confirming that feature/encoding decisions matter more than hyperparameter search at this data size.

## Final model

```python
CatBoostRegressor(
    iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    cat_features=['product_category_clean', 'fat_content', 'store_size',
                  'store_location_tier', 'store_format', 'product_code', 'store_code'],
    verbose=0, random_state=42
)
```

Numeric features: `product_weight_kg`, `shelf_visibility` (raw), `product_price`, `store_age_years`, `price_per_kg`, `visibility_x_price`

**CV RMSE: 1092.07 (± 18.61)** — trained via 5-fold cross-validation, refit on the full 6,818-row training set before generating test predictions.

## Notebook structure

1. Problem framing
2. Data loading & first look
3. Data quality audit
4. EDA
5. Cleaning & preprocessing
6. Feature engineering
7. Baseline model (mean, Ridge)
8. Iterative model comparison (RF, CatBoost variants, tuning)
9. Final model, test predictions & submission

## Setup

```bash
uv venv
uv pip install pandas numpy scikit-learn catboost lightgbm xgboost jupyter matplotlib seaborn joblib
```

## Known limitations

- Prediction max (~6,825) is well below the training max (~12,997) — the model does not predict any extreme high-sales outliers. Likely reflects that `test.csv` simply has no equivalently extreme rows, but this is unverified.
- Leaderboard RMSE not yet recorded — pending submission.
