################################################################################
# Setup & Working Directory 
################################################################################

from pathlib import Path
import os
import sys

PROJECT_NAME = "MALDIAlign"
cwd = Path().resolve()

# Walk upwards until we find the project folder
target = None
for parent in [cwd] + list(cwd.parents):
    if parent.name == PROJECT_NAME:
        target = parent
        break

# Change working directory to project root
if target is not None and target != cwd:
    os.chdir(target)

print("Working directory:", os.getcwd())

project_root = Path(os.getcwd())
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


################################################################################
# Imports 
################################################################################

import sys
import numpy as np
import pandas as pd

from datetime import datetime
from utils.load_config import load_config
from utils.load_data import load_pkl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


################################################################################
# Logging Setup 
################################################################################

# Create logs directory if missing
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = log_dir / f"logistic_baseline_{timestamp}.log"
results_file = log_dir / f"logistic_results_{timestamp}.csv"

print(f"Logging to: {log_file}")


################################################################################
# Data Loading 
################################################################################

cfg = load_config()
driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL"]
driams = load_pkl(driams_pkl)
data, label, meta = driams["data"], driams["label"], driams["meta"]
meta = pd.DataFrame.from_records(list(meta))


################################################################################
# Filter by Hospital 
################################################################################

filtered_data = {}
for hosp in meta["hospital"].unique():
    idx = np.where(meta["hospital"].values == hosp)[0]
    filtered_data[hosp] = {
        "data": data[idx],
        "label": label[idx],
        "meta": meta.iloc[idx]
    }

# Individual hospital datasets
dataA, labelA, metaA = filtered_data["DRIAMS_A"]["data"], filtered_data["DRIAMS_A"]["label"], filtered_data["DRIAMS_A"]["meta"]
dataB, labelB, metaB = filtered_data["DRIAMS_B"]["data"], filtered_data["DRIAMS_B"]["label"], filtered_data["DRIAMS_B"]["meta"]
dataC, labelC, metaC = filtered_data["DRIAMS_C"]["data"], filtered_data["DRIAMS_C"]["label"], filtered_data["DRIAMS_C"]["meta"]
dataD, labelD, metaD = filtered_data["DRIAMS_D"]["data"], filtered_data["DRIAMS_D"]["label"], filtered_data["DRIAMS_D"]["meta"]


################################################################################
# Split Train/Test (DRIAMS-A) 
################################################################################

X_train, _, y_train, _ = train_test_split(
    dataA,
    labelA,
    train_size=0.5, 
    stratify=labelA
    )

################################################################################
## Model Definition
################################################################################

# Define the pipeline
pipe_logistic = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=816)),
    ('lr', LogisticRegression(max_iter=1000))
])

# Define the hyperparameter grid
param_grid = {
    'lr__C': np.logspace(-3, 3, 10),
}

# Train 
grid_logistic = GridSearchCV(
    estimator=pipe_logistic,
    param_grid=param_grid,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

print(f"=== Logistic Regression Baseline Run ===")
print(f"Timestamp: {timestamp}")
print("Starting grid search...")

grid_logistic.fit(X_train, y_train)

print("\n=== Grid Search Finished ===")
print("Best parameters:", grid_logistic.best_params_)
print("Best balanced accuracy (CV):", grid_logistic.best_score_)
