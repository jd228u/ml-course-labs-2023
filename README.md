# Machine Learning Course Labs (2023)

This repository collects my lab work for a university Machine Learning course.
All solutions and analysis are my own unless explicitly marked as **course-provided skeleton code**.
Skeleton cells/functions were distributed by the course to help students focus on the ML concepts.

> **Academic integrity & reuse**
>
> These materials are for portfolio/learning. If you’re taking a similar course,
> follow your institution’s policies and **do not submit this work as your own**.

---

## How to run locally

```bash
# 1) set up a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt

# 3) launch notebooks
jupyter notebook
```
> **Tip:** To keep notebook diffs small, you can strip outputs before committing:
> ```bash
> pip install nbstripout && nbstripout --install
> ```

> **Data**
>
> Large datasets are **not** committed. Each lab either downloads its data from a public URL
> or instructs where to place your local copy. See the table below for per‑lab notes.

---

## Labs overview

| Lab | Topic (what you practice) | Key methods / libraries | Dataset & source |
|---|---|---|---|
| **1** | Binary classification, EDA, k-NN, metrics | `pandas`, `numpy`, `seaborn`, `scikit-learn` (KNeighbors, confusion matrix) | **Wisconsin Breast Cancer** CSV — course mirror |
| **2** | Decision trees & model capacity (tuning `max_depth`), stratified splits | `scikit-learn` `DecisionTreeClassifier`, **(optional)** `graphviz` for tree viz | **Covertype (normalized subset)** CSV — course mirror |
| **3** | Logistic regression with polynomial features, proper train/val/test split | `scikit-learn` `LogisticRegression`, `PolynomialFeatures` | **Wisconsin Breast Cancer** CSV — course mirror |
| **4** | Comparing linear classifiers, cross-validation, ROC/AUC, imputing & scaling | `SGDClassifier`, `GaussianNB`, `StandardScaler`, `SimpleImputer`, ROC/AUC | **Wisconsin Breast Cancer** CSV — course mirror |
| **5** | Regression with SGD, feature prep, MAE evaluation | `scikit-learn` `SGDRegressor`, `train_test_split`, `matplotlib` | **Housing** CSV — public mirror |
| **6** | **Transfer learning** on images (CIFAR‑100), EfficientNetB0, fine‑tuning | `tensorflow.keras` (EfficientNetB0, GAP, Dense), `scikit-learn` split | **CIFAR‑100** via `keras.datasets` |
| **7** | **Ensembles** on Fashion‑MNIST: RandomForest, ExtraTrees, AdaBoost | `RandomForestClassifier`, `ExtraTreesClassifier`, `AdaBoostClassifier` | **Fashion‑MNIST** CSVs — course mirror |
| **8** | **Gaussian Process Regression** on synthetic data; kernels & uncertainty | `GaussianProcessRegressor` with `RBF`/`Matern`, `matplotlib` | Synthetic (generated) |
| **9** | **Unsupervised learning**: PCA for dimensionality reduction + K‑Means | `PCA`, `KMeans`, `MinMaxScaler`/`StandardScaler`, `seaborn` | **Customer Personality Analysis** (Kaggle) |
| **10** | **End‑to‑end classification** with messy data: imputing, encoding, pipelines, model comparison + CV | `ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `StratifiedKFold` with LR, NB, KNN, RF | `train.csv` / `test.csv` — course mirror |
| **11** | **Time‑series regression** with custom loss; compare KNN, Linear, Random Forest, **XGBoost**; forecast plot on last 5k rows | `LinearRegression`, `KNeighborsRegressor`, `RandomForestRegressor`, `XGBRegressor`; weighted SSE; `matplotlib` | `train.csv.gz` / `train.csv` — course mirror (**do not shuffle**) |

> **Where skeleton code appears**  
> Several labs include small blocks of **course‑provided skeleton code** (setup cells, function stubs, or task outlines) to ensure consistency (e.g., fixed splits, helpers, or seed handling). I keep that attribution in the code comments/markdown. All substantive implementations and analysis are my own.

---

## Data locations & notes

- **Breast Cancer** CSV: course mirror (used in Labs 1, 3, 4)
- **Covertype (normalized subset)** CSV: course mirror (Lab 2)
- **Housing** CSV: public mirror (Lab 5)
- **Fashion‑MNIST** CSVs: course mirror (Lab 7)
- **CIFAR‑100**: loaded via `tensorflow.keras.datasets` (Lab 6)
- **Customer Personality Analysis**: Kaggle zip → `marketing_campaign.csv` (Lab 9)
- **Time‑series `train.csv.gz` / `train.csv`**: course mirror (Lab 11). **Do not shuffle** before splitting; the notebooks show how to slice head/tail for train/validation.

If a URL ever changes, use the notebook comments—they show exactly how each file is loaded.

---

## Environment & extras

Base dependencies are listed in `requirements.txt`.

- **Graphviz (Lab 2, tree visualization)**: install both the Python `graphviz` package **and** the system Graphviz tool (`dot`).
- **TensorFlow (Lab 6 only)**: install if you plan to run that lab locally.
- **XGBoost (Lab 11)**: required for `XGBRegressor`.

---

## Repository structure

```
.
├─ Lab_1/  …  Lab_11/
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## Attribution

- Small portions of some notebooks (setup cells, stubs, and task scaffolding) are **course‑provided skeleton code**
- All original solution code, analysis, and commentary are mine.
- If you are a rights holder and prefer that skeleton fragments be removed, please contact me and I’ll adjust promptly.
