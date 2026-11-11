import json
import textwrap
import os
from pathlib import Path
import random
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.config_loader as config_loader

BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_INPUT_DIR = BASE_DIR / "data" / "input" / "knowledge"/ "conversion"
KNOWLEDGE_OUTPUT_DIR = BASE_DIR / "data" / "output" / "knowledge"/ "conversion"
  
INCOMPLETE_KNOWLEDGE= KNOWLEDGE_INPUT_DIR / "incomplete_knowledge.json"
"""path al file contenente i dati di conoscenza incompleti"""

COMPLETE_KNOWLEDGE= KNOWLEDGE_OUTPUT_DIR /"complete_knowledge.json"
"""path al file contenente i dati di conoscenza completi"""

def _read_json(file_path):
    """Legge il contenuto di un file json e lo trasforma in oggetti python
    
    Args:
        file_path: path al file json di configurazione da leggere
    
    Returns:
        l'oggetto python dottenuto dalla decodifica dal file json
    
    Raises:
        OSError: Se il file non può essere aperto o letto.
        json.JSONDecodeError: Se il file non contiene un JSON valido.
    """
    file_path = Path(file_path)
    
    with file_path.open("r", encoding="utf-8-sig") as file:
        print(f"file'{file_path} letto correttamente.")
        return json.load(file)

def save_knowledge(data, filename):
    """Salva il dataset in un file JSON sovrascrivendo un eventuale file già presente

    Se le directory che compongono filename non esistono, le crea.
    Se il file non esiste, lo crea

    Args:
        data (list):  lista contenente le conoscenze da salvare.
        filename (str): Percorso e nome del file JSON di destinazione.

    Returns:
        None

    """
    directory = os.path.dirname(filename)

    # Crea le directory mancanti, se necessario.
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Apre il file in modalità scrittura ("w").
    # Se esiste già, il suo contenuto viene cancellato.
    # Se non esiste, viene creato.
    with open(filename, "w", encoding="utf-8-sig") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"Knowledge salvata in {filename}: {len(data)} elementi.")

def main():
    configuration = config_loader.load_configuration()
    incomplete_data = _read_json(INCOMPLETE_KNOWLEDGE)
    data=[]

    for i in range(len(incomplete_data)):
        try:
            knowledge = incomplete_data[i]

            # Considera esclusivamente le conoscenze benevoli.
            if knowledge["kind"] != "benevolo":
                continue

            goal_id = knowledge["goal_id"]

            # Cerca i nodi della topologia aventi questo goal tra i goal associati.
            candidate_nodes = [
                node for node in configuration["topology"]["nodes"]
                if goal_id in node["node_goals"]
            ]

            # Verifica che esista almeno un nodo compatibile.
            #altrimenti passa alla conoscenza successiva
            if not candidate_nodes:
                raise ValueError(
                    f"Nessun nodo associato al goal {goal_id}"
                )

            # Seleziona casualmente un nodo tra quelli compatibili.
            selected_node = random.choice(candidate_nodes)

            # Costruisce la conoscenza nel formato richiesto.
            complete_knowledge = {
                "node_id": selected_node["node_id"],
                "goal_id": goal_id,

                "context": {
                    "scenario": knowledge["scenario"],
                    "measured_parameters": knowledge["measured_parameters"]
                },

                "action": {
                    "manipulated_parameters": knowledge["manipulated_parameters"],
                    "behavior": knowledge["behavior"],
                    "confidence": knowledge["confidence"]
                }
            }

            # Aggiunge la conoscenza alla lista finale.
            data.append(complete_knowledge)

            
        except Exception as exc:
            print(f"Record {i} ignorato: {exc}")
            continue

    save_knowledge(data,COMPLETE_KNOWLEDGE )


if __name__ == "__main__":
    main()