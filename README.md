# Comment Category Prediction

**NLP + Machine Learning pipeline** to classify user comments into 4 categories (general, identity/hate, political, violent/threatening) on a dataset of 198,000 rows. Built as a Kaggle competition submission.

---

## What I built

End-to-end ML pipeline covering the full data science workflow — from raw data to a competition-ready submission:

- **EDA** — Uncovered severe class imbalance (20.9x ratio), identified that three identity columns go missing as a block (73.4% of rows), and surfaced key vocabulary signals per class via TF-IDF analysis
- **Feature Engineering** — Combined TF-IDF text features with structured signals (upvotes, downvotes, platform flags, emoticons, identity columns) and hand-crafted text features (comment length, exclamation count, uppercase ratio)
- **Model Progression** — Baseline → Logistic Regression → Random Forest → **LightGBM with hyperparameter tuning**
- **Class Imbalance Handling** — Applied `class_weight='balanced'` and `is_unbalance=True` to avoid the trap of a dummy model scoring ~58% by always predicting the majority class
- **Evaluation** — Accuracy, classification reports, confusion matrices, and multi-class ROC curves (AUC per label)

## Key Results

| Model | Accuracy |
|---|---|
| Baseline (Dummy) | 0.5766 |
| Logistic Regression | — |
| Random Forest | — |
| LightGBM (untuned) | — |
| **LightGBM (tuned) ★** | **Best** |

> LightGBM's leaf-wise tree growth and gradient boosting on hard examples outperformed both linear and bagging approaches.

## Technical Stack

`Python` · `scikit-learn` · `LightGBM` · `TF-IDF` · `pandas` · `matplotlib` · `seaborn`

## Notable Findings

- **`if_2`** (a hidden platform signal) was the single most predictive numeric feature — separating Label 0 (mean ~4.9) from Labels 1–3 (mean ~12)
- **Label 3** (violent/threatening) was the easiest to classify — short comments with unambiguous vocabulary (*kill, shoot, death*)
- **Labels 0 and 2** were the hardest pair — political and general commentary share heavy vocabulary overlap, requiring gradient boosting to tease apart

## Files

| File | Description |
|---|---|
| `notebook.ipynb` | Full pipeline: EDA → preprocessing → modeling → submission |

---

*Dataset: [Comment Category Prediction Challenge](https://www.kaggle.com/) · Metric: Accuracy*
