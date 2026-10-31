"""Carica e valida i file di consfigurazione dell'applicazione

Il modulo definisce i path dei file di configurazione json, i quali sono attesi in  ``data/input/config``relativo alla root del progetto.
Errori emersi nel caricamento o validazione dei file sono caricati nel file di log, ma le relative eccezioni sono propogate al chiamante
"""


import json
from pathlib import Path
import logging
import math

BASE_DIR = Path(__file__).resolve().parents[2]

#:Directory contenente i file di configurazione
CONFIG_DIR = BASE_DIR / "data" / "input" / "config"

#:Mapping tra gli identificativi dei parametri di configurazione e i filename dei file che descrivono tali parametri
CONFIG_FILES = {
    "topology": "topology.json",
    "simulation_parameters": "simulation_parameters.json",
    "analysis_parameters": "analysis_parameters.json",
}

#: path al file di configurazione della topologia
TOPOLOGY_FILE = CONFIG_DIR / CONFIG_FILES["topology"]

#: path al file di configurazione dei parametri di simulazione
SIMULATION_PARAMETERS_FILE = CONFIG_DIR / CONFIG_FILES["simulation_parameters"]

#: path al file di configurazione dei parametri di analisi
ANALYSIS_PARAMETERS_FILE = CONFIG_DIR / CONFIG_FILES["analysis_parameters"]


logger = logging.getLogger(__name__)


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
    
    with file_path.open("r", encoding="utf-8") as file:
        logger.info(f"file'{file_path} letto correttamente.")
        return json.load(file)



def _validate_topology(topology):
    """Valida il formato e i valori della topologia 
    
    FORMATO ATTESO: dizionario con la seguente struttura
    "network_id": stringa non vuota 
    "network_name": stringa non vuota
    "network_description": stringa non vuota
    "network_goals": lista non vuota di dizionari con la seguente struttura- "goal_id": stringa non vuota, "description": stringa non vuota
    "nodes": lista di dizionari con la seguente struttura- "node_id": stringa non vuota,"node_name": stringa non vuota, "node_goals": lista non vuota di stringhe non vuote
    "links": lista di dizionari con la seguente struttura- "link_id":stringa non vuota, "source": stringa non vuota,"target": stringa non vuota.
     node_id deve essere un valore unico tra i nodi;
     link_id deve essere un valore unico tra i nodi;
     goal_id deve essere un valore unico tra i goal;
     "source" e "target" devono avere un valore che esiste in "nodo_id";
    "source" e "target" non possono avere lo stesso valore;
    "node_goals"deve essere formata da elementi che hanno un valore che esiste in "goal_id"

    Args:
        topology: oggetto python che descrive la topologia della rete

    Returns:
        None. La funzione termina normalmente se la topologia è valida
    
    Raises:
        ValueError: se la topologia ha un formato diverso da quello atteso o non rispetta i prerequesiti definiti
    """

    #Verifica che la topologia sia stata descritta come un dizionario,
    #altrimenti solleva un'eccezione 
    if not isinstance(topology, dict):
        raise ValueError("La topologia deve essere un dizionario JSON.")

    #Verifica che nel dizionario ci siano tutte le chiavi attese il cui valore è di tipo stringa,
    #verifica che il valore per ciascuna di queste chiavi non sia una stringa vuota
    #altrimenti solleva un'eccezione
    for field in ("network_id", "network_name", "network_description"):
        if not isinstance(topology[field], str) or not topology.get(field).strip():
            raise ValueError(
                f"il campo '{field}' deve essere una stringa non vuota."
            )

    #Verifica che nel dizionario ci siano tutte le chiavi attese il cui valore è di tipo lista,
    #verifica che il valore per ciascuna di queste chiavi non sia una lista vuota
    #altrimenti solleva un'eccezione
    for field in ("network_goals", "nodes", "links"):
        if not isinstance(topology.get(field), list) or not topology.get(field):
            raise ValueError(f"il campo '{field}' deve essere una lista non vuota.")

    goals=topology.get("network_goals")
    goal_ids = set()

    #Verifica che ogni elemento della lista sia un dizionario
    #altrimenti solleva un'eccezione
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            raise ValueError(
                f"il goal all'indice {index} deve essere un dizionario."
            )

        #Verifica che nel dizionario ci siano tutte le chiavi attese il cui valore è di tipo stringa,
        #verifica che il valore per ciascuna di queste chiavi non sia una stringa vuota
        #altrimenti solleva un'eccezione
        for field in ("goal_id", "description"):
            value = goal.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"il campo'{field}' in goal all'indice {index} "
                    "deve essere una stringa non vuota."
                )

        goal_id = goal["goal_id"]

        #Verifica che non esistano duplicati in goal_id, altrimenti solleva un'eccezion
        if goal_id in goal_ids:
            raise ValueError(f"goal ID duplicato: '{goal_id}'.")

        goal_ids.add(goal_id)


    nodes = topology.get("nodes")
    node_ids = set()
    
    #Verifica che ogni elemento della lista sia un dizionario
    #altrimenti solleva un'eccezione
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(
                f"il nodo all'indice {index} deve essere un dizionario."
            )
        #Verifica che nel dizionario ci siano tutte le chiavi attese il cui valore è di tipo stringa,
        #verifica che il valore per ciascuna di queste chiavi non sia una stringa vuota
        #altrimenti solleva un'eccezione
        for field in ("node_id", "node_name"):
            value = node.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"il campo'{field}' al nodo {index} "
                    "deve essere una stringa non vuota."
                )

        node_id = node["node_id"]

        #Verifica che non esistano duplicati in node_id, altrimenti solleva un'eccezione
        if node_id in node_ids:
            raise ValueError(f"node ID duplicato: '{node_id}'.")

        node_ids.add(node_id)


        node_goals = node.get("node_goals")

        #Verifica che "node_goals" sia una lista non vuota, altrimenti solleva un'eccezione
        if not isinstance(node_goals, list) or not node_goals:
            raise ValueError(
                f"'node_goals' nel nodo '{node_id}' "
                "deve essere una lista non vuota."
            )

        #Verifica che ogni goal in tale lista sia una stringa non vuota, altrimenti solleva un'eccezione
        for goal_id in node_goals:
            if not isinstance(goal_id, str) or not goal_id.strip():
                raise ValueError(
                    f"ogni goal nel nodo  '{node_id}' "
                    "deve essere una stringa non vuota."
                )
            #Verifica che ogni goal in tale lista esista nella lista di network_goals 
            if goal_id not in goal_ids:
                raise ValueError(
                    f"il '{node_id}' fa riferimento a un goal ID sconosciuto: "
                    f"'{goal_id}'."
                )


    links = topology.get("links")
    link_ids=set()
    connected_nodes = set()
    
    #Verifica che ogni elemento della lista di link sia un dizionario
    #altrimenti solleva un'eccezione
    for index, link in enumerate(links):
        if not isinstance(link, dict):
            raise ValueError(
                f"il link ad indice {index} deve essere un dizionario."
            )

        #Verifica che nel dizionario ci siano tutte le chiavi attese il cui valore è di tipo stringa,
        #verifica che il valore per ciascuna di queste chiavi non sia una stringa vuota
        #altrimenti solleva un'eccezione
        for field in ("link_id","source","target" ):
            if not isinstance(link[field], str) or not link.get(field).strip():
                raise ValueError(
                    f" '{field}' al link {index} deve essere una stringa non vuota."
            )
        link_id=link.get("link_id")
        source = link.get("source")
        target = link.get("target")

        #Verifica che non esistano duplicati in link_id, altrimenti solleva un'eccezione
        if link_id in link_ids:
            raise ValueError(f"link ID duplicato: '{link_id}'.")

        link_ids.add(link_id)

        #Verifica che ogni node_id usato per source o target esista nella lista di node_id 
        if source not in node_ids:
            raise ValueError(
                f"Link {index} referenzia un source node sconosciuto:'{source}'."
            )

        if target not in node_ids:
            raise ValueError(
                f"Link {index}  referenzia un target node sconosciuto:'{target}'."
            )

        #Verifica che source e target non coincidano  
        if source == target:
            raise ValueError(
                f"Link {index} non può connettere '{source}' con se stesso."
            )

        connected_nodes.add(source)
        connected_nodes.add(target)

    #Verifica che ogni nodo sia collegato ad almeno un link, altrimenti solleva un'eccezione
    for node_id in node_ids:
        if node_id not in connected_nodes:
            raise ValueError(
                f"il nodo '{node_id}' non è connesso ad alcun link."
            )
    
    logger.info(f"topologia validata")





def _validate_simulation_parameters(param):
    """Valida il formato e i valori dei parametri di simulazione

    FORMATO ATTESO: dizionario con chiavi "measured_parameters" e "manipulated_parameters",
    i cui valori corrispondenti siano liste di dizionari non vuote.
    Ogni elemento della lista associata a measured_parameters deve avere due chiavi associate a valori di tipo stringa, non vuuote: ("name", "description", "unit")
    e tre chiavi associati a valori numerici ("min","max","delta") tc min<max, entrambi maggiori di zero, e 0<delta<=100
    Ogni elemento della lista associata a manipulated_parameters deve avere due chiavi associate a valori di tipo stringa, non vuuote ("name", "description", "unit")
    e tre chiavi associati a valori numerici ("min","max","weight") tc min<max, tutti maggiori di zero,

    Args:
        param: oggetto python che descrive i parametri di simulazione della rete

    Returns:
        None. La funzione termina normalmente se i parametri di simulazione sono validi
    
    Raises:
        ValueError: se i parametri di simulazione hanno un formato diverso da quello atteso o non rispettano i prerequesiti definiti

    """
    #Verifica che i parametri siano stati descritti come un dizionario,
    #altrimenti solleva un'eccezione 
    if not isinstance(param, dict):
        raise ValueError(
            "I parametri devono essere un dizionario JSON."
        )

    #Verifica che nel dizionario ci siano ambo le chiavi attese
    #altrimenti solleva un'eccezione
    for key in ("measured_parameters", "manipulated_parameters"):
        if key not in param:
            raise ValueError(
                f"Il campo '{key}' è obbligatorio."
            )

        #Verifica che il valore associato alla chiave corrente sia una lista non vuota
        #altrimenti solleva un'eccezione
        if not isinstance(param[key], list)or not param[key]:
            raise ValueError(
                f"Il campo '{key}' deve essere una lista non vuota."
            )

        for index, item in enumerate(param[key]):
            #Verifica che ogni elemento della lista sia un dizionario
            #altrimenti solleva un'eccezione
            if not isinstance(item, dict):
                raise ValueError(
                    f"L'elemento {index} di '{key}' deve essere un dizionario."
                )

            for field in ("name", "description", "unit"):
                #Verifica che il valore per tali chiavi sia una stringa e sia una stringa non vuota
                # altrimenti solleva un'eccezione
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere una stringa non vuota."
                    )

            
            for field in ("min", "max"):
                #Verifica che il valore per le chiavi "min", "max","weight" sia un valore numerico
                #altrimenti solleva un'eccezione
                #ATT: i booleani sono un sottotipo di int

                value = item.get(field)

                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                ):
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere numerico."
                    )
                # Verifica che il valore sia finito
                #altrimenti solleva un'eccezione
                if not math.isfinite(value):
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere finito."
                    )

                # Verifica che il valore sia positivo
                #altrimenti solleva un'eccezione
                if value <= 0:
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere strettamente maggiore di 0."
                    )


            #Verifica che min<max, altrimenti solleva un'eccezione
            if item["min"] >= item["max"]:
                raise ValueError(
                    f"Il valore 'min' non può essere maggiore di 'max' "
                    f"nell'elemento {index} di '{key}'."
                )

            # Verifica dei parametri specifici di ciascuna lista
            if key == "measured_parameters":
                field = "delta"
                value = item.get(field)

                # Verifica che delta sia numerico e non booleano
                #altrimenti solleva un'eccezione
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere numerico."
                    )

                # Verifica che delta sia finito
                #altrimenti solleva un'eccezione
                if not math.isfinite(value):
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere finito."
                    )

                # Verifica 0 < delta <= 100
                #altrimenti solleva un'eccezione
                if not 0 < value <= 100:
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere maggiore di 0 "
                        f"e minore o uguale a 100."
                    )

            elif key == "manipulated_parameters":
                field = "weight"
                value = item.get(field)

                # Verifica che weight sia numerico e non booleano
                #altrimenti solleva un'eccezione
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere numerico."
                    )

                # Verifica che weight sia finito
                #altrimenti solleva un'eccezione
                if not math.isfinite(value):
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere finito."
                    )
                
                # Verifica che weight sia strettamente maggiore di zero
                #altrimenti solleva un'eccezione
                if value <= 0:
                    raise ValueError(
                        f"Il campo '{field}' dell'elemento {index} "
                        f"di '{key}' deve essere strettamente maggiore di 0."
                    )

    logger.info(f"parametri di simulazione validati")

    
def _validate_analysis_parameters(param):
    """Valida il formato e i valori dei parametri di analisi
    
    FORMATO ATTESO:dizionario con la seguente struttura
    "measured_parameters_analysis": a cui è associato un dizionario con le seguenti chiavi:
        -k_NN: intero positvo
        -max_distance: float positivo
    "TTL": a cui è associato un intero postivo
    "audit_parameters":  a cui è associato un dizionario con le seguenti chiavi:
        -damage_score: intero positivo compreso tra 0 e 100
        -threshold: intero postivo compreso tra 0 e 100
        -merit_score: intero positivo compreso tra 0 e 100

    Args:
        param: oggetto python che descrive i parametri di analisi della rete

    Returns:
        None. La funzione termina normalmente se i parametri di analisi sono validi
    
    Raises:
        ValueError: se la i parametri di analisi hanno un formato diverso da quello atteso o non rispettano i prerequesiti definiti

    """
    # Verifica che i parametri siano stati descritti come un dizionario
    #altrimenti solleva un'eccezione 
    if not isinstance(param, dict):
        raise ValueError(
            "I parametri di analisi devono essere un dizionario JSON."
        )

    # Verifica che nel dizionario siano presenti tutte le chiavi attese
    #altrimenti solleva un'eccezione
    for key in (
        "measured_parameters_analysis",
        "TTL",
        "audit_parameters"
    ):
        if key not in param:
            raise ValueError(
                f"Il campo '{key}' è obbligatorio."
            )

    # Verifica che measured_parameters_analysis sia un dizionario
    #altrimenti solleva un'eccezione
    measured = param["measured_parameters_analysis"]
    if not isinstance(measured, dict):
        raise ValueError(
            "Il campo 'measured_parameters_analysis' "
            "deve essere un dizionario."
        )

    # Verifica che siano presenti le chiavi obbligatorie
    #altrimenti solleva un'eccezione
    for field in ("k_NN", "max_distance"):
        if field not in measured:
            raise ValueError(
                f"Il campo '{field}' di 'measured_parameters_analysis' "
                f"è obbligatorio."
            )

    # Verifica che k_NN sia un intero strettamente positivo
    value = measured["k_NN"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "Il campo 'k_NN' di 'measured_parameters_analysis' "
            "deve essere un intero."
        )

    if value <= 0:
        raise ValueError(
            "Il campo 'k_NN' di 'measured_parameters_analysis' "
            "deve essere strettamente maggiore di 0."
        )

    # Verifica che max_distance sia numerico, finito e positivo
    #altrimenti solleva un'eccezione
    value = measured["max_distance"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "Il campo 'max_distance' di 'measured_parameters_analysis' "
            "deve essere numerico."
        )

    if not math.isfinite(value):
        raise ValueError(
            "Il campo 'max_distance' di 'measured_parameters_analysis' "
            "deve essere finito."
        )

    if value <= 0:
        raise ValueError(
            "Il campo 'max_distance' di 'measured_parameters_analysis' "
            "deve essere strettamente maggiore di 0."
        )

    # Verifica che TTL sia un intero strettamente positivo
    #altrimenti solleva un'eccezione
    value = param["TTL"]

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "Il campo 'TTL' deve essere un intero."
        )

    if value <= 0:
        raise ValueError(
            "Il campo 'TTL' deve essere strettamente maggiore di 0."
        )

    # Verifica che audit_parameters sia un dizionario
    #altrimenti solleva un'eccezione
    audit = param["audit_parameters"]

    if not isinstance(audit, dict):
        raise ValueError(
            "Il campo 'audit_parameters' deve essere un dizionario."
        )

    # Verifica che siano presenti tutte le chiavi obbligatorie
    #altrimenti solleva un'eccezione
    for field in ("damage_score", "threshold", "merit_score"):
        if field not in audit:
            raise ValueError(
                f"Il campo '{field}' di 'audit_parameters' "
                f"è obbligatorio."
            )

    # Verifica che i punteggi siano interi compresi tra 0 e 100
    #altrimenti solleva un'eccezione
    for field in ("damage_score", "threshold", "merit_score"):
        value = audit[field]

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Il campo '{field}' di 'audit_parameters' "
                f"deve essere un intero."
            )

        if not 0 < value <= 100:
            raise ValueError(
                f"Il campo '{field}' di 'audit_parameters' "
                f"deve essere compreso tra 0 e 100."
            )

    logger.info("parametri di analisi validati")


def load_configuration():
    """Legge e valida tutti i file di configurazione.
    
    Returns:
        un dizionario contenente le chiavi "topology", "simulation_parameters" e "analysis_parameters",
        ciascuna mappata all'oggetto python letto dal corrispondente dile file di configurazione

    Raises:
        OSError: Se uno dei file di configurazione non può essere aperto o letto.
        json.JSONDecodeError: Se uno dei file di configurazione non contiene un JSON valido.
        ValueError: se la topologia,i parametri di simulazione o di analisi hanno un formato diverso da quello atteso o non rispettano i prerequesiti definiti
    
    Notes:
        Un'eventuale eccezione è registrata sul file di log prima di essere propagata al chiamante
    """
    
    try: 
        topology = _read_json(TOPOLOGY_FILE)
        simulation_param = _read_json(SIMULATION_PARAMETERS_FILE)
        analysis_param = _read_json(ANALYSIS_PARAMETERS_FILE)

        _validate_topology(topology) 
        _validate_simulation_parameters(simulation_param)
        _validate_analysis_parameters(analysis_param)

    except Exception as exc:
        logger.exception("%s", exc)
        raise 

    logger.info("configurazione caricata correttamente.")
    return {
        "topology": topology,
        "simulation_parameters": simulation_param,
        "analysis_parameters": analysis_param
    }

def load_config_file( file_path):
    """Carica il file di configurazione indicato e restituisce il valore caricato e la chiave a cui corrisponde nella configurzione completa.
    
    Args:
        file_path: path al file json di configurazione da leggere
    
    Returns:
        una tupla (chiave, valore), tc 
        chiave= "topology"o "simulation_parameters" o "analysis_parameters" a seconda del nome del file di configurazione
        valore= l'oggetto json letto dal file di configurazione

    Raises:
        OSError: Se il file di configurazione non può essere aperto o letto.
        json.JSONDecodeError: Se il file di configurazione non contiene un JSON valido.
        ValueError: se il contenuto del file ha un formato diverso da quello atteso o non rispetta i prerequesiti definiti
    
    Notes:
        Un'eventuale eccezione è registrata sul file di log prima di essere propagata al chiamante
    """

    try:
        value = _read_json(file_path)

        file_name = Path(file_path).name
        match file_name:
            case "topology.json":
                _validate_topology(value)
                key = "topology"

            case "simulation_parameters.json":
                _validate_simulation_parameters(value)
                key = "simulation_parameters"
        
            case "analysis_parameters.json":


                _validate_analysis_parameters(value)
                key = "analysis_parameters"

            case _:
                raise ValueError(
                    f"File di configurazione non riconosciuto: {file_name}"
                )
    except Exception as exc:
        logger.exception("%s", exc)
        raise 

    logger.info(f"{key} caricata/i correttamente.")    
    return key, value
