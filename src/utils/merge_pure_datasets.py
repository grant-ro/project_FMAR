
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
TRAINING_INPUT_DIR = BASE_DIR / "data" / "input" / "training" / "merge"
TRAINING_OUTPUT_DIR = BASE_DIR / "data" / "output" / "training" /"merge"
  
PURE_DATASET= TRAINING_INPUT_DIR / "pure_dataset"
PURE_DATASET_MERGED=TRAINING_OUTPUT_DIR / "pure_dataset_merged.json"

def merge_dataset():
    # Carica il dataset principale
    if PURE_DATASET.exists() and os.path.getsize(PURE_DATASET) > 0:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            dataset_finale = json.load(f)
    else:
        dataset_finale = []

    # Scorre tutti i file JSON nella cartella
    for file_json in TRAINING_INPUT_DIR.glob("*.json"):

        # Evita di leggere nuovamente il file principale
        if file_json.name == PURE_DATASET.name:
            continue

        print(f"Lettura: {file_json.name}")

        try:
            with open(file_json, "r", encoding="utf-8-sig") as f:
                dati = json.load(f)

            # Aggiunge i dati al dataset principale
            if isinstance(dati, list):
                dataset_finale.extend(dati)
            else:
                dataset_finale.append(dati)

        except (json.JSONDecodeError, OSError) as e:
            print(f"Errore nel file {file_json.name}: {e}")

    # Salva il dataset aggiornato
    with open(PURE_DATASET_MERGED, "w", encoding="utf-8-sig") as f:
        json.dump(
            dataset_finale,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\nUnione completata!")
    print(f"Dataset finale: {PURE_DATASET_MERGED}")
    print(f"Numero totale di elementi: {len(dataset_finale)}")


if __name__ == "__main__":
    merge_dataset()
