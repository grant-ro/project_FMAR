
import json
import os
from pathlib import Path
from core.simulation.slm_handler import get_SLM_name, generate_insight, evaluate_insight
import core.config_loader as config_loader

BASE_DIR = Path(__file__).resolve().parents[1]
EVALUATION_INPUT_DIR = BASE_DIR / "data" / "input" / "evaluation" 
EVALUATION_OUTPUT_DIR = BASE_DIR / "data" / "output" / "evaluation" 

EVALUATION_INPUT= EVALUATION_INPUT_DIR / "config"
EVALUATION_OUTPUT =EVALUATION_OUTPUT_DIR / "evaluation_results.json"


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

def evaluate_model(scenario,configuration, model):
    """Valuta il modello 

    Args:
        scenario: Dizionario avente struttura:
            "scenario_description": la descrizione dello scenario generato
            "measured_parameters": lista di dizionari aventi 
            -chiave= nome del parametro misurato,
            -valore= valore generato dal modello linguistico 
        configuration: dizionario che descrive la configurazione della rete, avente
            -topology, che contiene la chiave "network_description", "network_goals"(che per ogni goal prevede una chiave "description"), "nodes"(che per ogni goal prevede una chiave "node_id")
            -simulation_parameters, che contiene la chiave "manipulated_parameters. questo per ogni parametro manipolabile contiene 
                informazioni, quali: nome, description, unità, min e max
        model: modello da valutare

    Returns:
        dict: Dizionario JSON contenente:
            - behavior: descrizione del comportamento del nodo.
            - manipulated_parameters: dizionario con i valori proposti per ciascun parametro manipolabile 
            -confidence:il livello di fiducia del nodo nella soluzione proposta (valore tra 0 e 100).

    """

    network_description=configuration["topology"]["network_description"]
    manipulated_parameters=configuration["simulation_parameters"]["manipulated_parameters"]

    #Selezione il primo goal tra i goal della rete e ne estrae la descrizione
    selected_goal = configuration["topology"]["network_goals"][0]
    goal_desc = selected_goal["description"]

    #Selezione il primo nodo tra i nodi della rete 
    source_id = configuration["topology"]["nodes"][0]["node_id"]

    malicious_insight= generate_insight(
        kind="malevolo",
        network_description=network_description, 
        data=scenario,
        manipulated_parameters=manipulated_parameters,
        goal_description="Congestionare la rete",
        model_name=model
    )

    benevolent_insight= generate_insight(
        kind="benevolo",
        network_description=network_description, 
        data=scenario,
        manipulated_parameters=manipulated_parameters,
        goal_description=goal_desc,
        model_name=model
    )

    malicious_evaluation= evaluate_insight(
        source_id=source_id,
        network_description=data["topology"]["network_description"],
        data=scenario,
        manipulated_parameters=simulation_parameters["manipulated_parameters"],
        goal_description=goal_desc,
        insight=malicious_insight,
        model_name=model,
    )

    benevolent_evaluation= evaluate_insight(
        source_id=source_id,
        network_description=data["topology"]["network_description"],
        data=scenario,
        manipulated_parameters=simulation_parameters["manipulated_parameters"],
        goal_description=goal_desc,
        insight=benevolent_insight,
        model_name=model,
    )

    results={
        "benevolent": benevolent_insight,
        "malicious":malicious_insight,
        "evaluation of benevolent": benevolent_evaluation,
        "evaluation of malicious": malicious_evaluation
    }

def save_results(scenario. result, filename):
    """Salva lo scenario e i risultati dell'evaluation del modello in un file txt

    Se le directory che compongono filename non esistono, le crea.
    Se il file non esiste, lo crea

    Args:
        scenario(dict): dizionario contenente lo scenario di test
        result (dict): dizionario contenente i risultati da salvare.
        filename (str): Percorso e nome del file JSON di destinazione.

    Returns:
        None

    """
    
    # Crea le directory mancanti
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Scrive i dati nel file di testo
    with open(filename, "w", encoding="utf-8") as file:

        file.write("SCENARIO\n")
        for key, value in scenario.items():
            file.write(f"{key}: {value}\n")

        file.write("\nRISULTATI\n")
        for key, value in result.items():
            file.write(f"{key}: {value}\n")

    print(f"scenario e risultati di test salvati con successo)

def main():
    model_name = input("Inserisci il nome del modello da testare: \n Le opzioni sono: llama-3.2-1b, llama-traffic, qwen-rete").strip()
    configuration = config_loader.load_configuration()
    scenario = _read_json(EVALUATION_INPUT)
    results = evaluate_model(scenario, configuration,model_name)
    save_dataset(scenario, results, EVALUATION_OUTPUT)

if __name__ == "__main__":
    main()