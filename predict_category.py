"""Interaktivno testiranje modela: unesi naziv proizvoda, dobij kategoriju.

Pokretanje:  python predict_category.py
Izlaz iz programa: ukucaj 'exit' ili 'q'.
"""
import pickle

MODEL_PATH = "models/model.pkl"


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Klasifikator proizvoda. Ukucaj naziv proizvoda ('exit' za izlaz).")
    while True:
        title = input("\nNaziv proizvoda: ").strip()
        if title.lower() in {"exit", "q", "quit"}:
            break
        if not title:
            continue
        print("Predvidjena kategorija:", model.predict([title.lower()])[0])


if __name__ == "__main__":
    main()
