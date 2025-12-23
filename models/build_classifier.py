from sklearn.ensemble import RandomForestClassifier

def build_classifier(classifier_type: str, classifier_cfg: dict):
    if classifier_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=classifier_cfg["n_estimators"],
            max_depth=classifier_cfg["max_depth"],
            class_weight=classifier_cfg["class_weight"],
            n_jobs=classifier_cfg["n_jobs"],
            random_state=classifier_cfg["random_state"]
        )
    else:
        return None
        # TODO: Include more classifiers
    return clf
    