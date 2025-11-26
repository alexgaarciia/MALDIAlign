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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


################################################################################
# Logging Setup 
################################################################################

# Create logs directory if missing
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = log_dir / f"svm_rbf_baseline_{timestamp}.log"
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

# Select only 20% of DRIAMS-A because of computational cost
dataA_sub, _, labelA_sub, _ = train_test_split(
    dataA,
    labelA,
    train_size=0.2,
    stratify=labelA,
    random_state=42
)


################################################################################
# Split Train/Test (DRIAMS-A Subset) 
################################################################################

X_train, _, y_train, _ = train_test_split(
    dataA_sub,
    labelA_sub,
    test_size=0.2,   
    stratify=labelA_sub,
    shuffle=True,
    random_state=42
)

################################################################################
## Model Definition
################################################################################

# Define the pipeline
pipe_svc_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', class_weight='balanced'))
])

# Define the hyperparameter grid
param_grid_svc_rbf = {
    'svm__C': np.logspace(-3, 3, 10),
    'svm__gamma': np.logspace(-3, 3, 10) / X_train.shape[1]
}

# Train 
grid_svc_rbf = GridSearchCV(
    estimator=pipe_svc_rbf,
    param_grid=param_grid_svc_rbf,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

print("=== RBF SVM Baseline Run ===")
print(f"Timestamp: {timestamp}")
print("Starting grid search...")

grid_svc_rbf.fit(X_train, y_train)

print("\n=== Grid Search Finished ===")
print("Best parameters:", grid_svc_rbf.best_params_)
print("Best balanced accuracy (CV):", grid_svc_rbf.best_score_)
