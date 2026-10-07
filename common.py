"""Zajednicke funkcije: ucitavanje i ciscenje podataka."""
import pandas as pd

# Mapiranje nekonzistentnih oznaka kategorija na jedinstvene nazive
LABEL_FIXES = {
    "fridge": "Fridges",
    "cpu": "CPUs",
    "mobile phone": "Mobile Phones",
}


def load_and_clean(path="data/products.csv"):
    """Ucitava products.csv i vraca ociscen DataFrame sa kolonama title, category.

    Koraci: uklanjanje razmaka iz naziva kolona, brisanje redova bez naslova
    ili kategorije, ujednacavanje naziva kategorija, mala slova, brisanje duplikata.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Product Title", "Category Label"]).copy()

    df["category"] = df["Category Label"].str.strip()
    df["category"] = df["category"].apply(lambda x: LABEL_FIXES.get(x.lower(), x))
    df["title"] = df["Product Title"].str.lower().str.strip()
    df = df[df["title"].str.len() > 0]
    df = df.drop_duplicates(subset=["title", "category"])
    return df[["title", "category"]].reset_index(drop=True)
