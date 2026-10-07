"""Treniranje modela za klasifikaciju proizvoda po nazivu.

Pokretanje:  python train_model.py
Izlaz:       models/model.pkl, reports/confusion_matrix.png, reports/classification_report.txt
"""
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from common import load_and_clean

DATA_PATH = "data/products.csv"
MODEL_PATH = "models/model.pkl"


def build_pipeline():
    """TF-IDF (rijeci + karakteri) -> LinearSVC."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("tfidf", features), ("clf", LinearSVC(C=1.0, class_weight="balanced"))])


def main():
    df = load_and_clean(DATA_PATH)
    print(f"Podataka nakon ciscenja: {len(df)}, kategorija: {df['category'].nunique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["title"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(report)

    with open("reports/classification_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=45, ax=ax, cmap="Blues")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=120)

    # Finalni model: treniran na CELOM skupu podataka
    model.fit(df["title"], df["category"])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model sacuvan u {MODEL_PATH}")


if __name__ == "__main__":
    main()
