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
from utils.config import load_config
from utils.data import load_pkl

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


################################################################################
# Logging Setup 
################################################################################

# Create logs directory if missing
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = log_dir / f"rf_baseline_{timestamp}.log"
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

dataA_sub, _, labelA_sub, _ = train_test_split(
    dataA,
    labelA,
    train_size=0.5,
    stratify=labelA,
    shuffle=True,
    random_state=42
)

# Apply normalization: scale each spectrum to [0, 1]
X_min = dataA_sub.min(axis=1, keepdims=True)
X_max = dataA_sub.max(axis=1, keepdims=True)
dataA_sub_norm = (dataA_sub - X_min) / (X_max - X_min + 1e-8)


################################################################################
## Model Definition
################################################################################

# Define the pipeline
pipe_rf = Pipeline([
    ("rf", RandomForestClassifier(
        class_weight="balanced_subsample", 
        n_jobs=-1,
        random_state=42
    ))
])

# Define the hyperparameter grid
param_grid_rf = [
    {
        "rf__max_depth": [None, 20, 40],
        "rf__n_estimators": [200, 400, 800],
    }
]

# Train 
grid_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1
)

sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

print("=== Random Forest Baseline Run ===")
print(f"Timestamp: {timestamp}")
print("Starting grid search...")

grid_rf.fit(dataA_sub_norm, labelA_sub)

print("\n=== Grid Search Finished ===")
print("Best parameters:", grid_rf.best_params_)
print("Best Balanced Accuracy (CV):", grid_rf.best_score_)
