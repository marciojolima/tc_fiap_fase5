from sklearn.ensemble import RandomForestClassifier


def build_baseline_model(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def build_challenger_model(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)
