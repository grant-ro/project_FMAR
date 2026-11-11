
import threading
import requests
import os
import json
import random
from core.simulation.slm_handler import get_SLM_name, generate_network_scenario, generate_insight, evaluate_insight
from core.simulation.local_knowledge_handler import init_local_knowledge, search_lk, add_knowledge
from core.simulation.dlt_handler import init_dlt, search_dlt, add_record, reset_dlt, is_dlt_setup
from core.simulation.audit import init_audit, get_node_known_nature, get_node_true_nature, audit_reasoning, audit_trust_evaluation
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_local_knowledge = None
"""Lista che associa un dizionario contenente la conoscenza locale del nodo per ogni nodo della topologia."""

_adjacency = None
"""Dizionario che associa per ogni nodo al suo node_id la lista dei node_id dei nodi adiacenti ad esso."""

_audit_state=False
"""Bool che rappresenta lo stato dell'audit e della sua conoscenza: True: configurato, False: non configurato"""

_reasoning_mode= None
"""Stringa che rappresenta il livello di intelligenza da usare nella simulazione, e quindi i server da utilizzare"""

_scenario_generation_server= None
"""Nome del server da usare per la generazione dello scenario"""

_insight_generation_server = None
"""Nome del server da usare per la generazione dell'insight"""

_insight_evaluation_server= None
"""Nome del server da usare per la valutazione dell'insight"""

_audit_server= None
"""Nome del server che usa l'audit"""

BASE_DIR = Path(__file__).resolve().parents[3]
TRAINING_OUTPUT_DIR = BASE_DIR / "data" / "output" / "training"
PURE_DATASET= TRAINING_OUTPUT_DIR / "pure_dataset.json"
"""path al file contenente i dati puri di training  """

KNOWLEDGE_INPUT_DIR =BASE_DIR / "data" / "input" / "knowledge"
INITIAL_KNOWLEDGE= KNOWLEDGE_INPUT_DIR/"knowledge.json"
"""path al file contenente la conoscenza iniziale da caricare"""

def _init_reasoning_mode(mode):
    """
    Inizializza la tipologia di reasoning da usare nella simulazione

    Args:
        reasoning_mode= Stringa che indica la tipologia di reasoning mode da usare nella simulazione e qiuindi a quali server affidarsi

    Returns:
        None
    
    """
    global _scenario_generation_server
    global _insight_generation_server
    global _insight_evaluation_server
    global _audit_server 

    match mode:
        case "dumb":
            _scenario_generation_server = get_SLM_name("llama-3.2-1b")
            _insight_generation_server = get_SLM_name("llama-3.2-1b")
            _insight_evaluation_server =  get_SLM_name("llama-3.2-1b")
            _audit_server= get_SLM_name("llama-3.2-1b")

        case "smart":
            _scenario_generation_server = get_SLM_name("qwen-rete")
            _insight_generation_server = get_SLM_name("qwen-rete")
            _insight_evaluation_server = get_SLM_name("qwen-rete")
            _audit_server= get_SLM_name("qwen-rete")

        case "traffic":
            _scenario_generation_server = get_SLM_name("qwen-rete")
            _insight_generation_server = get_SLM_name("llama-traffic")
            _insight_evaluation_server = get_SLM_name("llama-traffic")
            _audit_server= get_SLM_name("qwen-rete")

        case _:
            raise ValueError(f"Reasoning mode non valido: {mode}")




def _build_adjacency(topology):
    """Costruisce la lista di adiacenza a partire dalla topologia della rete.

    Args:
        topology (dict): Struttura contenente la topologia della rete, composta da:
            - nodes (list[dict]): Lista dei nodi. Ogni nodo è un dizionario composto da
                -node_id: identificativo del nodo.
            - links (list[dict]): Lista dei collegamenti. Ogni collegamento è un dizionario composto da
                -source: node_id del nodo da cui ha origine il collegamento
                -target: node_id del nodo su cui termina il collegamento

    Returns:
        dict[str, list]: Dizionario che associa per ogni nodo al suo node_id la lista (senza duplicati)
            dei node_id dei nodi adiacenti ad esso secondo quanto descritto in topology. 
            Sono considerati adiacenti ad un nodo tutti i nodi con cui esso è collegato 
            mediante almeno un link, in quanto source o target di quest'ultimo. 
    """

    # L'utilizzo dei set impedisce la presenza di nodi duplicati nelle liste di nodi adiacenti.
    # come nel caso di due nodi collegati in ambo le direzioni
    adjacency = {
        node["node_id"]: set()
        for node in topology["nodes"]
    }

    for link in topology["links"]:
        source = link["source"]
        target = link["target"]

        #aggiunge il nodo target nella lista dei nodi adiacenti al nodo source
        adjacency[source].add(target)

        #aggiunge il nodo source nella lista dei nodi adiacenti al nodo target
        adjacency[target].add(source)
    
    # Converte gli insiemi in liste per restituire la struttura prevista.
    return {
        node_id: list(adjacents)
        for node_id, adjacents in adjacency.items()
    }

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

def _load_knowledge(param_configuration):
    """legge e carica la conoscenza contenuta in INITIAL_KNOWLEDGE nella conoscenza locale dei nodi e nel DLT

    Si aspetta che INITIAL_KNOWLEDGE contenga una lista di dizionari, aventi tutta la seguente struttura:
        node_id: id del nodo che ha generato la conoscenza corrente
        goal_id: goal rispetto al quale ha generato tale conoscenza
        context: dizionario avente chiavi scenario e measured_parameters
        action: dizionario avente chiavi manipultaed_parameters, behavior e confidence

    Args:
        param_configuration(dict): configurazione utilizzata per la costruzione dei vettori. 
                Deve contenere per ogni measured parameter chiavi "weight", "min","max" 
    
    Returns:
        None

    """
    initial_knowledge = _read_json(INITIAL_KNOWLEDGE)

    if not initial_knowledge:
        logger.warning("INITIAL_KNOWLEDGE è vuoto: nessuna conoscenza da caricare.")
        return

    logger.info(
        "Caricamento di %d elementi di conoscenza da INITIAL_KNOWLEDGE.",
        len(initial_knowledge)
    )

    for new_record in initial_knowledge:
        node_id = new_record["node_id"]
        goal_id = new_record["goal_id"]

        #Caricamento della conoscenza corrente nel DLT
        add_record(new_record, param_configuration)

        #Caricamento della conoscenza corrente nella conoscenza locale 
        # del nodo che ha prodotto questa conoscenza per il goal con cui l'ha prodotta
        local_knowledge_node = next(
            (node for node in local_knowledge if node["node_id"] == node_id),
            None
        )

        if local_knowledge_node is None:
            logger.warning(
                "Nodo %s non trovato nella conoscenza locale. Record non caricato in LK.",
                node_id
            )
            continue

        if goal_id not in local_knowledge_node:
            logger.warning(
                "Goal %s non trovato per il nodo %s. Record non caricato in LK.",
                goal_id, node_id
            )
            continue

        add_knowledge(
            local_knowledge_node[goal_id],
            new_record,
            param_configuration
        )

    logger.info("Caricamento della conoscenza iniziale completato.")
    


def _setup_simulation(configuration, nodes_true_nature, reasoning_mode):
    """Inizializza la conoscenza locale, la struttura di adiacenza della rete, 
        la conoscenza relativa alla natura dei nodi dell'audit, la reasoning mode. 
    
    Args:
        configuration: Dizionario contenente la configurazione della simulazione, inclusa la topologia della rete.
        nodes_true_nature= Dizionario che associa per ogni nodo al suo node_id la stringa che rappresenta la sua reale natura (benevolo o malevolo)
        reasoning_mode= Stringa che indica la tipologia di reasoning mode da usare nella simulazione e qiuindi a quali server affidarsi

    Returns:
        None.
    
    """
    global _local_knowledge, _adjacency, _audit_state

    _local_knowledge = init_local_knowledge(
        configuration["topology"]["nodes"]
    )
    
    _adjacency = _build_adjacency(
        configuration["topology"]
    )

    init_audit(audit_parameters=configuration["analysis_parameters"]["audit_parameters"], nodes_true_nature= nodes_true_nature)
    _init_reasoning_mode(reasoning_mode)
    _audit_state=True
    init_dlt()

    logger.info("Simulazione inizializzata.")


def reset_simulation():
    """Resetta lo stato interno della simulazione: 
    azzera la conoscenza locale, la struttura di adiacenza della rete, 
    la conoscenza relativa alla natura dei nodi della'audit
    impostando tali variabili globali a None/False 

    Args:
        None.

    Returns:
        None.

    """
    global _local_knowledge, _adjacency, _nodes_known_nature, _nodes_true_nature

    _local_knowledge = None
    _adjacency = None
    _audit_state= False
    _reasoning_mode= None
    reset_DLT()

    logger.info("Stato interno della simulazione resettato.")

def _calculate_insight(results, knowledge, node_id="nodo sorgente"):
    """
    Calcola un insight a partire dalle azioni associate ai casi
    simili individuati nella conoscenza locale.

    Se viene trovata una corrispondenza esatta, restituisce
    direttamente l'azione associata al record corrispondente.

    In caso contrario, calcola una media pesata delle azioni
    dei casi simili, attribuendo a ciascuno un'importanza
     inversamente proporzionale alla loro distanza.

    Args:
        results (list[dict]): risultati della ricerca di similarità.
            Ogni elemento contiene:
                - index (int): indice del record nella conoscenza.
                - distance (float): distanza dalla query.

        knowledge (dict): base di conoscenza contenente i record
            e le relative azioni, accessibili tramite knowledge["data"].

        node_id (str, optional): identificativo del nodo associato
            alla ricerca.

    Returns:
        dict: insight calcolata, contenente i parametri dell'azione
            e i rispettivi valori.

        None: se non sono disponibili risultati.
    """

    if not results:
        logger.warning(
            "Nodo %s: nessun risultato disponibile per il calcolo dell'insight.",
            node_id
        )
        return None

    if results[0]["distance"] == 0:

        index = results[0]["index"]
        insight = knowledge["data"][index]["action"].copy()
        logger.info(
            "Nodo %s: trovata una corrispondenza esatta nella conoscenza locale.",
            node_id
        )

        return insight

    weighted_values = {}
    total_weight = 0.0

    for result in results:
        index = result["index"]
        distance = result["distance"]

        action = knowledge["data"][index]["action"]
        weight = 1.0 / distance
        total_weight += weight

        for param, value in action.items():
            weighted_values[param] = (
                weighted_values.get(param, 0.0) + weight * value
            )

    insight = {
        param: value / total_weight
        for param, value in weighted_values.items()
    }

    logger.info(
        "Nodo %s: %sinsight calcolata tramite media pesata di %d casi simili.",
        node_id,
        "comparison " if node_id == "nodo sorgente" else "",
        len(results)
    )

    return insight

def _benevolent_node_reasoning(local_knowledge,network_description, scenario, goal_description, simulation_parameters, k, max_distance ):
    """Genera l'insight  di un nodo benevolo della rete.
    
    Args:
        local_knowledge=la conoscenza locale di riferimento del nodo per lo specifico goal del reasoning
        network_description: Descrizione della natura della rete
        scenario: Dizionario con scenario_description e measured_parameters.
        simulation_parameters: dizionario con seguente struttura:
            -manipulalted_parameters=Lista di dizionari contenenti i parametri modificabili, ciascuno con nome, descrizione, unità di misura, min, max 
            -measured_parameters= Lista di dizionari contenenti i parametri modificabili, ciascuno con nome, descrizione, unità di misura, min, max e weight,
        goal_description: Obiettivo del nodo. Deve essere specificato per i nodi benevoli e deve essere None per quelli malevoli.
        k (int): numero massimo di elementi simili da cercare nella conoscenzalocale del nodo.
        max_distance (float): massima_distanza che gli elementi simili ricercati nella conoscenza locale del nodo deveono possedere rispetto allo scenario di riferimento
        
        
    Returns:
        Dizionario JSON contenente:
            -flag= True, se l'insight è stata prodotta affidandosi all'slm, False altrimenti;
            -insight= un dizionario avente la seguente struttura se flag=true
                - behavior: descrizione del comportamento del nodo.
                - manipulated_parameters: dizionario con i valori proposti per ciascun parametro manipolabile 
                -confidence:il livello di fiducia del nodo nella soluzione proposta (valore tra 0 e 100).
            avente, altrimenti, la seguente struttura:
                -per ogni parametro in simulation_parameters["manipulated_parameters"] un valore 
    Raises:
        ValueError: Se il modello linguistico produce un risultato non conforme al formato atteso oppure con valori non validi 
        Altri tipi di eccezione se fallisce la comunicazione HTTP con il server del modello.
    """
    flag=True

    #Cerca di generare l'insight nel contesto dello scenario corrente
    # affidandosi alla proprio conoscenza locale

    #Cerca nella conoscenza locale situazioni affrontate in passato con un 
    #contesto sufficientemente simile
    logger.info(
            "Nodo sorgente: valutazione basata sulla conoscenza locale.",
        )

    results= search_lk(query=scenario["measured_parameters"], k=k, max_distance=max_distance, search_space=local_knowledge, param_configuration=simulation_parameters["measured_parameters"])    
    if results:
        flag=False
        #Calcola l'insight a partire dalle action effettuate nelle situazioni simili in passato
        insight = _calculate_insight(
            results,
            local_knowledge
        ) 
    else:

        #Cerca di generare l'insight nel contesto dello scenario corrente
        # affidandosi alla conoscenza contenuta nel DLT 

        #Cerca nel DLT situazioni affrontate in passato con un 
        #contesto sufficientemente simile
        logger.info(
            "Nodo sorgente: valutazione basata sul DLT.",
        )
        results= search_dlt(query=scenario["measured_parameters"], k=k, max_distance=max_distance, param_configuration=simulation_parameters["measured_parameters"]) 
        if results:
            flag=False
            #Calcola l'insight a partire dalle action effettuate nelle situazioni simili in passato
            insight = _calculate_insight(
                results,
                local_knowledge
            )  

        else:
            #Genera l'insight di natura benevola nel contesto dello scenario corrente 
            #affidandosi a un SLM
            logger.info(
                "Nessun caso simile trovato: generazione insight tramite SLM."
            )

            insight= generate_insight(
                kind="benevolo",
                network_description=network_description, 
                data=scenario,
                manipulated_parameters=simulation_parameters["manipulated_parameters"],
                goal_description=goal_description,
                model_name=_insight_generation_server
            )

    return {
        "flag": flag,
        "insight": insight
    }

def _malicious_node_reasoning(network_description, scenario, manipulated_parameters):
    """Genera l'insight  di un nodo malevolo della rete.
    
    Args:
        network_description: Descrizione della natura della rete
        scenario: Dizionario con scenario_description e measured_parameters.
        manipulated_parameters: Lista di dizionari contenenti i parametri modificabili, ciascuno con nome, descrizione, unità di misura, min e max.
        
    Returns:
        Dizionario JSON contenente:
            -flag= True altrimenti;
            -insight= un dizionario avente la seguente struttura 
                - behavior: descrizione del comportamento del nodo.
                - manipulated_parameters: dizionario con i valori proposti per ciascun parametro manipolabile 
                -confidence:il livello di fiducia del nodo nella soluzione proposta (valore tra 0 e 100).
    Raises:
        ValueError: Se il modello linguistico produce un risultato non conforme al formato atteso oppure con valori non validi 
        Altri tipi di eccezione se fallisce la comunicazione HTTP con il server del modello.
    """

    #Genera l'insight di natura malevola nel contesto dello scenario precedentemente generato,
    #affidandosi a un SLM
    insight= generate_insight(
        kind="malevolo",
        network_description=network_description, 
        data=scenario,
        manipulated_parameters=manipulated_parameters,
        goal_description="Congestionare la rete",
        model_name=_insight_generation_server
    )

    return {
        "flag": True,
        "insight": insight
    }

def _get_nodes_within_k_steps(adjacency, source_id, k):
    """Restituisce i nodi raggiungibili dal nodo sorgente entro k passi.

    Args:
        adjacency (dict [str, list]): Dizionario che associa per ogni nodo al suo node_id la lista dei node_id dei nodi adiacenti ad esso.
        source_id: Identificativo del nodo sorgente.
        k (int): Numero massimo di passi consentiti per raggiungere i nodi vicini al nodo sorgente.
        
    Notes:
        k deve essere un intero positivo.
        adjacency deve possedere una voce associata a source_id
        adjacency deve possedere una voce associata a ogni nodo raggiunto mediante l'esplorazione
        La lista dei nodi raggiungibili dal nodo sorgente entro k passi è ottenuta mediante un'esplorazione in ampiezza della topologia


    Returns:
        list: Lista degli identificativi dei nodi raggiungibili dalla sorgente con un numero di passi compreso tra 1 e k, 
            Il nodo sorgente non è incluso nella lista dei nodi raggiungibili.

    """


    visited = {source_id}       #lista dei nodi già visitati
    queue = deque([source_id])  #lista dei nodi da visitare
    result = []                 #lista dei nodi raggiungibili entro k salti

    for _ in range(k):
        # Se non ci sono altri nodi da esplorare, interrompe la ricerca.
        if not queue:
            break

        for _ in range(len(queue)):
            #effettua un'esplorazione in ampiezza (BFS, Breadth-First Search).
            current = queue.popleft()

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append(neighbor)

    return result

def _insight_evaluation(node_id, data):
    """Valuta l'insight prodotta dal nodo sorgente 

    Args:
        node_id: identificativo del nodo che esegue questa funzione
        data (dict): struttura dati così composta:
            -source_id: identificativo del nodo che ha generato l'insight da esaminare
            -goal_id:identificativo del goal di riferimento
            -topology: dizionario che contiene le info relative alla topologia della rete, così composto:
                -network_description: descrizione della rete
                -nodes:lista di dizionari,
                -goals:lista di dizionari, ciascuno dei quali possiede una chiave 'id', e una chiave 'description'
            -simulation_parameters:dizionario che contiene le info relative ai parametri di simulazione, così composto:
                -measured_parameters: lista di dizionari, ciascuno dei quali composto dai campi: name, description, unit, min, max, weight
                -manipulated_parameters: lista di dizionari, ciascuno dei quali composto dai campi: name, description, unit, min, max, delta
            -k (int): numero massimo di elementi simili da cercare nella conoscenzalocale del nodo.
            -max_distance (float): massima_distanza che gli elementi simili ricercati nella conoscenza locale del nodo deveono possedere rispetto allo scenario di riferimento
            -scenario(dict): Dizionario con scenario_description e measured_parameters che definiscono il contesto dell'insight
            -insight: un dizionario avente la seguente struttura se flag=true
                - behavior: descrizione del comportamento del nodo.
                - manipulated_parameters: dizionario con i valori proposti per ciascun parametro manipolabile 
                -confidence:il livello di fiducia del nodo nella soluzione proposta (valore tra 0 e 100).
            
    Returns:
        bool: True se il nodo approva l'insight in analisi, False  altrimenti
    """

    logger.info(
        "Nodo %s: avvio valutazione insight generata dal nodo %s.",
        node_id, data["source_id"]
    )
    node_true_nature= get_node_true_nature(node_id)
    insight = data["insight"]
    goal_id = data["goal_id"]
    scenario = data["scenario"]
    simulation_parameters= data["simulation_parameters"]
    k=data["k"]
    max_distance=data["max_distance"]
    

    #PRIMO CONTROLLO
    # Controlla che ogni parametro manipolabili valorizzato dall'insight, 
    # rientri nei limiti definiti nella configurazione dei parametri  manipolabili 
    allowed_parameters = {
        parameter["name"]: parameter
        for parameter in data["simulation_parameters"]["manipulated_parameters"]
    }

    for parameter_name, value in insight["manipulated_parameters"].items():
        parameter_config = allowed_parameters[parameter_name]
        min_value = parameter_config["min"]
        max_value = parameter_config["max"]

        # Interrompe immediatamente se un parametro  è fuori intervallo.
        if value < min_value or value > max_value:
            if node_true_nature=="benevolo":
                logger.info(
                    "Nodo %s:   INSIGHT RIFIUTATA, parametro '%s' fuori dai limiti consentiti "
                    "[%s, %s], valore proposto: %s.",
                    node_id, parameter_name, min_value, max_value, value
                )
                return False
            else:
                logger.info(
                    "Nodo %s:   INSIGHT APPROVATA, NODO MALEVOLO, parametro '%s' fuori dai limiti consentiti "
                    "[%s, %s], valore proposto: %s.",
                    node_id, parameter_name, min_value, max_value, value
                )
                return False


    #SECONDO CONTROLLO
    node_knowledge = next(
        (
            node for node in _local_knowledge
            if node["nodo_id"] == node_id
        ),
        None
    )

    goal_specific_knowledge=node_knowledge[goal_id]
    if goal_specific_knowledge["data"]:
        #Cerca nella conoscenza locale situazioni affrontate in passato con un 
        #contesto sufficientemente simile
        logger.info(
            "Nodo %s: valutazione basata sulla conoscenza locale per il goal %s.",
            node_id, data["goal_id"]
        )

        results= search_lk(query=scenario["measured_parameters"], k=k, max_distance=max_distance, search_space=goal_specific_knowledge["context_vectors"], param_configuration=simulation_parameters["measured_parameters"]) 
        if results:
            #Calcola l'insight a partire dalle action effettuate nelle situazioni simili in passato
            comparison_insight = _calculate_insight(
                results,
                goal_specific_knowledge,
                node_id
            )

            #Se l'insight calcolata è abbastanza simile a quella da valutare
            # la valutazione termina in modo positivo, altrimenti l'analisi continua
            parameters_config = simulation_parameters["manipulated_parameters"]
            all_within_delta = True

            for parameter in parameters_config:
                name = parameter["name"]
                delta = parameter["delta"]

                proposed_value = insight[name]
                comparison_value = comparison_insight[name]

                if comparison_value == 0:
                    difference_percentage = (
                        0.0 if proposed_value == 0 else float("inf")
                    )
                else:
                    difference_percentage = (
                        abs(proposed_value - comparison_value)
                        / abs(comparison_value)
                    * 100
                )

                if difference_percentage >= delta:
                    all_within_delta = False
                    break

            if all_within_delta:
                if node_true_nature=="benevolo":
                    logger.info(
                        "Nodo %s: INSIGHT APPROVATA tramite confronto con la conoscenza locale.",
                        node_id
                    )
                    return True
                else:
                    logger.info(
                        "Nodo %s: INSIGHT RIFIUTATA: NODO MALEVOLO, tramite confronto con la conoscenza locale.",
                        node_id
                    ) 

                    return False
            

    #TERZO CONTROLLO
    #Infine il node agent prova ad affidarsi all'SLM
    logger.info(
        "Nodo %s: conoscenza locale insufficiente; avvio valutazione tramite SLM.",
        node_id
    )
    topology=data["topology"]
    for goal in topology["network_goals"]:
            if goal["goal_id"] == goal_id:
                goal_desc = goal["description"]
                break

    evaluation= evaluate_insight(
        source_id=data["source_id"],
        network_description=data["topology"]["network_description"],
        data=scenario,
        manipulated_parameters=simulation_parameters["manipulated_parameters"],
        goal_description=goal_desc,
        insight=insight,
        model_name=_insight_evaluation_server,
    )
    if node_true_nature=="malevolo":
        evaluation= not evaluation
        logger.info(
            "Nodo %s: MALEVOLO, valutazione SLM completata. Esito: %s.",
            node_id, evaluation
        )

    logger.info(
        "Nodo %s: valutazione SLM completata. Esito: %s.",
        node_id, evaluation
    )
    return evaluation

def _distribute_and_wait(adjacency, source_id, goal_id,topology, k, data, worker):
    """Distribuisce un'attività ai nodi raggiungibili entro k passi e attende
     i risultati di tale attività fino al raggiungimento di una maggioranza.

    Args:
        adjacency (dict [str, list]): Dizionario che associa per ogni nodo al suo node_id la lista dei node_id dei nodi adiacenti ad esso.
        source_id: Identificativo del nodo sorgente da cui avviare la ricerca dei nodi raggiungibili.
        goal_id: identificativo del goal rispetto al quale sta operando la sorgente
        topology: topologia dei nodi della rete, che ad ogni nodo associa una lista dei goal a cui esso aderisce.
        k (int): Numero massimo di passi entro cui individuare i nodi a cui distribuire l'attività.
        data: Dati da passare a ciascun worker per l'esecuzione dell'attività.
        worker (callable): Funzione da eseguire in parallelo su ciascun nodo raggiungibile. 
            Deve ricevere come argomenti l'identificativo del nodo e data.

    Notes: worker deve restituire un valore booleano, True oppure False, che rappresenti l'esito dell'attività.

    Returns:
        tuple: Tupla contenente:
            - results (dict): Dizionario che associa al node_id di ogni nodo il risultato restituito dal worker,
                                per le esecuzioni completate senza eccezioni.
            - errors (dict): Dizionario che associa al node_id di ogni nodo che ha terminato la proporia attività con un'eccezione
                              il messaggio dell'eccezione verificatasi durante l'esecuzione.
            - majority (bool or None): True se viene raggiunta una maggioranza di risultati positivi, 
                                        False se viene raggiunta una maggioranza di risultati negativi,
                                        None se non viene raggiunta alcuna maggioranza.

    """
   
    
    #Individua i nodi raggiungibili dal nodo sorgente entro k passi.
    nodes = _get_nodes_within_k_steps(adjacency, source_id, k)

    logger.info(
        "Distribuzione attività dal nodo %s: %d nodi raggiungibili entro %d salti.",
        source_id, len(nodes), k
    )

    results = {}
    errors = {}

    executor = ThreadPoolExecutor(max_workers=None)

    # Avvia un'attività per ciascun nodo individuato, 
    # la cui nota natura sia benevola e il cui goal sia il medesimo del nodo sorgente
    # executor.submit() pianifica l'esecuzione e restituisce un Future,
    # un oggetto che rappresenta l'attività e permette di recuperarne il risultato.
    futures = {
        executor.submit(worker, node, deepcopy(data)): node
        for node in nodes
        if (
            get_node_known_nature(node) == "benevolo"
            and goal_id in next(
            (item["node_goals"] for item in topology
             if item["node_id"] == node),
            []
            )
        )
    }

    logger.info(
        "Nodo sorgente %s: avviate %d valutazioni presso nodi benevoli "
        "che condividono il goal %s.",
        source_id, len(futures), goal_id
    )

    majority = None

    try:
        # Elabora i Future man mano che la relativa attività termina
        for future in as_completed(futures):
            node = futures[future]

            try:
                # Se l'attività è terminata correttamente, 
                # associa il risultato al nodo corrispondente.
                results[node] = future.result()

            except Exception as exc:
                #altrimenti salva il messaggio di errore generato dall'attività
                # e l'associa al nodo corrispondente.
                errors[node] = str(exc)
                logger.info(
                    "Nodo %s: valutazione terminata con errore: %s.",
                    node, exc
                )
                continue

            # Conta i risultati positivi e negativi
            positive_count = sum(
                1 for result in results.values()
                if result is True
            )

            negative_count = sum(
                1 for result in results.values()
                if result is False
            )

            total_tasks = len(futures) - len(errors)

            # Maggioranza positiva
            if positive_count > total_tasks / 2:
                majority = True
                logger.info(
                    "QUORUM: Raggiunta maggioranza positiva."
                )
                break

            # Maggioranza negativa
            if negative_count > total_tasks / 2:
                majority = False
                logger.info(
                    "QUORUM: Raggiunta maggioranza negativa."
                )
                break

    finally:

        if majority is not None:
            # Se è stata raggiunta una maggioranza, 
            # tenta di annullare le attività non ancora avviate.
            # e non attende la conclusione delle attività ancora in esecuzione.
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            # Se non è stata raggiunta una maggioranza
            # (se sono stati elaborati tutti i Future (attesa preventiva)
            # o è stata sollevata un'eccezione non gestita),
            # attende la conclusione di tutte le attività pianificate.
            executor.shutdown(wait=True)

    logger.info(
        "Distribuzione completata per il nodo %s: %d risultati, "
        "%d errori, esito quorum=%s.",
        source_id, len(results), len(errors), majority
    )
    return results, errors, majority

def run_short_simulation(configuration, data):
    """Esegue una simulazione breve.

    Args:
        configuration (dict): Configurazione utilizzata dalla simulazione.
        data (dict):  Dati aggiuntivi per eseguire la simulazione. Contiene
             - node_id (str): Identificativo del nodo coinvolto nella simulazione. Deve esistere in configuration
             - node_nature (str): Natura del nodo. Valori ammessi: "benevolo" o "malevolo"
             - nodes_true_nature (dict): dizionario che associa ad ogni nodo della topologia la sua reale natura. Valori ammessi per ogni nodo: "benevolo" o "malevolo"
             -reasoning_mode (str): tipologia di ragionamento desiderato. Valori ammessi "dumb" "medium"
            
        Any: Risultato della simulazione.
    """
    global _local_knowledge, _adjacency, _audit_state, _reasoning_node

    if _local_knowledge is None or _adjacency is None or _audit_state == False or _reasoning_mode is None or is_dlt_setup==False:
        _setup_simulation(configuration, data["nodes_true_nature"], data["reasoning_mode"])
    
    logger.info(
        "Avvio simulazione breve: nodo=%s, natura=%s.",
        data["node_id"], data["node_nature"]
    )

    #Genera uno scenario critico in cui deve operare il nodo selezionato
    scenario= generate_network_scenario(
        configuration["simulation_parameters"]["measured_parameters"],
        configuration["topology"]["network_description"],
        model_name=_scenario_generation_server
    )
    logger.info("Scenario di rete generato correttamente.")

    #Seleziona il nodo in configuration avente l'id specificato in data["node_id"]
    for current_node in configuration["topology"]["nodes"]:
        if current_node["node_id"] == data["node_id"]:
            node = current_node
            break

    #Seleziona randomicamente uno dei goal associato al nodo selezionato
    node_goals = node["node_goals"]
    selected_goal = random.choice(node_goals)
    logger.info(
        "Nodo %s: selezionato goal %s.",
        data["node_id"], selected_goal
    )

    logger.info(
        "Nodo %s: Avvio produzione insight.",
        data["node_id"]
    )
    simulation_start_time = time.perf_counter() #istante prima della generazione dell'insight

    #Estrae per il goal selezionato la sua descrizione
    for goal in configuration["topology"]["network_goals"]:
        if goal["goal_id"] == selected_goal:
            goal_desc = goal["description"]
            break

    #Estrae la conoscenza locale associata al nodo e al goal selezionati
    for node in _local_knowledge:
        if node["nodo_id"] == data["node_id"]:
            goal_specific_knowledge=node[selected_goal]
            break

    if data["node_nature"] == "benevolo":        

        #Produce un'insight benevola 
        insight=_benevolent_node_reasoning(
            local_knowledge=goal_specific_knowledge,
            network_description=configuration["topology"]["network_description"],
            scenario= scenario,
            goal_description=goal_desc,
            simulation_parameters= configuration["simulation_parameters"],
            k=configuration["analysis_parameters"]["measured_parameters_analysis"]["k_NN"],
            max_distance=configuration["analysis_parameters"]["measured_parameters_analysis"]["max_distance"]
        )
    else:
        #Produce un'insight malevola 
        insight=_malicious_node_reasoning(
            network_description=configuration["topology"]["network_description"],
            scenario= scenario,
            manipulated_parameters= configuration["simulation_parameters"]["manipulated_parameters"]
        )
    
    logger.info(
        "Nodo %s: generazione insight completata. Origine=%s.",
        data["node_id"],
        "conoscenza locale" if data["node_nature"] == "benevolo"
        and not insight["flag"] else "SLM"
        )

    if insight["flag"]==False:
        logger.info(
            "Nodo %s: insight recuperata dalla conoscenza locale; "
            "nessuna votazione richiesta.",
            data["node_id"]
        )
        simulation_end_time=time.perf_counter()  #istante immediatamente successivo alla produzione dell'insight, quorum non necessario

    else:
        #invoca il meccanismo del quorum, quindi trasmette l'insight prodotta ai nodi prossimi per una loro valutazione dell'insight
        logger.info(
            "Nodo %s: insight generata mediante slm; "
            "Votazione richiesta: avvio processo di quorum.",
            data["node_id"]
        )

        node_agent_data = {
            "source_id": data["node_id"],
            "goal_id": selected_goal,
            "topology": configuration["topology"],
            "simulation_parameters": configuration["simulation_parameters"],
            "k": configuration["analysis_parameters"]["measured_parameters_analysis"]["k_NN"],
            "max_distance": configuration["analysis_parameters"]["measured_parameters_analysis"]["max_distance"],
            "scenario": scenario,
            "insight": insight["insight"],
        }
        results, errors, majority= _distribute_and_wait(adjacency=_adjacency, source_id=data["node_id"],goal_id=selected_goal, topology=configuration["topology"]["nodes"], k=configuration["analysis_parameters"]["TTL"], data=node_agent_data , worker=_insight_evaluation)
        simulation_end_time=time.perf_counter()  #istante immediatamente successivo al quorum
        insight=insight["insight"]

        # se nessun altro nodo è raggiungibile in questo momento o non è stata raggiunta una maggioranza dei voti tra i nodi raggiungibili
        # si lascia produrre l'insight all'audit
        if not results or majority is None:
            logger.info(
                "Nodo %s: quorum non conclusivo; richiesta di generazione "
                "dell'insight all'audit.",
                data["node_id"]
            )
            try:
                insight= audit_reasoning(
                    network_description=configuration["topology"]["network_description"],
                    scenario= scenario,
                    goal_description=goal_desc,
                    manipulated_parameters=configuration["simulation_parameters"]["manipulated_parameters"],
                    model_name=get_SLM_name("llama-3.1-8b")
                )

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                logger.error(
                    "Connection error or timeout  durante audit_reasoning():%s. ",
                    str(e)
                )
                
                insight= audit_reasoning(
                    network_description=configuration["topology"]["network_description"],
                    scenario= scenario,
                    goal_description=goal_desc,
                    manipulated_parameters=configuration["simulation_parameters"]["manipulated_parameters"],
                    model_name=_audit_server
                )

        else: 
            #se è stata raggiunta una maggioranza tra i voti dei nodi raggiungibili
            #si notifica all'audit l'esito della votazione, affinchè esso possa provvedere ad aggiornare la natura dei nodi
            logger.info(
                "Nodo %s: quorum concluso con esito %s; aggiornamento della fiducia "
                "tramite audit.",
                data["node_id"], majority
            )
            
            audit_trust_evaluation(results=results,majority=majority,node_id=data["node_id"])
            if majority is True:
                if data["node_nature"] == "benevolo":
                    logger.info(
                        "ESITO_SIMULAZIONE: SUCCESSO, insight del nodo benevolo approvata ."
                    )
                else:
                    logger.info(
                        "ESITO_SIMULAZIONE: INSUCCESSO, insight del nodo malevolo approvata"
                    )
            else:
                if data["node_nature"] == "benevolo":
                    logger.info(
                        "ESITO_SIMULAZIONE: INSUCCESSO, insight del nodo benevolo rifiutata."
                    )
                else:
                    logger.info(
                        "ESITO_SIMULAZIONE: SUCCESSO, insight del nodo malevolo rifiutata."
                    )

        if majority is not False:
            new_data_lk = {
                "action": insight["manipulated_parameters"],
                "context":scenario["measured_parameters"]
                }

            logger.info(
                "Nodo %s: maggioranza positiva; aggiornamento della conoscenza locale "
                "per il goal %s.",
                data["node_id"], selected_goal
            )

            add_knowledge(local_knowledge=goal_specific_knowledge, new_data=new_data_lk, data_configuration=configuration["simulation_parameters"]["measured_parameters"])

            new_data_dlt={
                "node_id":data["node_id"],
                "goal_id":selected_goal,
                "context": {
                    "scenario":scenario["scenario_description"],
                    "measured_parameters":scenario["measured_parameters"]
                    },
                "action":{
                    "manipulated_parameters":insight["manipulated_parameters"],
                    "behavior":insight["behavior"],
                    "confidence":insight["confidence"]
                    }
            }

            logger.info(
                "Aggiornamento del dlt "
            )
            add_record(new_record=new_data_dlt,param_configuration=configuration["simulation_parameters"]["measured_parameters"])

    total_time =simulation_end_time- simulation_start_time
    logger.info(
    "TEMPI_SIMULAZIONE: %.3f s",
    total_time
)

    return


def run_long_simulation(configuration, stop_event: threading.Event):
    """Esegue una simulazione lunga con supporto all'arresto.

    Args:
        configuration (dict): Configurazione utilizzata dalla simulazione.
        stop_event (threading.Event): Evento utilizzato per richiedere
            l'interruzione della simulazione.

    Returns:
        Any: Risultato della simulazione.
    """

    logger.info("Avvio generazione dataset.")

    result = None

    while not stop_event.is_set():
        # Esegue un passo della simulazione.
        logger.info("Esecuzione di un nuovo passo della generazione del dataset.")
        


        # Esempio:
        # result = ...

        # Interrompere il ciclo quando la simulazione è terminata.
        break

    logger.info("Simulazione lunga terminata.")
    return result

def save_dataset(dataset, filename):
    """Salva il dataset in un file JSON, aggiungendo i nuovi esempi
    a quelli eventualmente già presenti.

    Se le directory che compongono filename non esistono, le crea.
    Se il file non esiste, lo crea

    Args:
        dataset (list): Dataset contenente gli esempi da salvare.
        filename (str): Percorso e nome del file JSON di destinazione.

    Returns:
        None

    """
    #Crea tutte le directory mancanti che compongono filename se non esistono già
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    

    if os.path.exists(filename)and os.path.getsize(filename) > 0:
        #se filename esiste già e non è vuoto, apre il file in lettura e ne carica il contenuto
        with open(filename, "r", encoding="utf-8") as file:
            existing_dataset = json.load(file)
    else:
        existing_dataset = []

    #aggiunge tutti gli elementi contenuti in dataset alla fine di existing_dataset
    existing_dataset.extend(dataset)    

    #apre filename in scrittura (se il file non esiste, lo crea)
    #e ne sovrascrive il contenuto con il nuovo dataset
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(existing_dataset, file, indent=4, ensure_ascii=False)


def run_dataset_generation(configuration, stop_event: threading.Event):
    """Esegue una simulazione lunga con supporto all'arresto.

    Args:
        configuration (dict): Configurazione utilizzata dalla simulazione.
        stop_event (threading.Event): Evento utilizzato per richiedere
            l'interruzione della simulazione.

    Returns:
        None
        
    """

    logger.info("Avvio generazione dataset.")

    dataset=[]

    mode="smart"
    _init_reasoning_mode(mode)

    network_description=configuration["topology"]["network_description"]
    manipulated_parameters=configuration["simulation_parameters"]["manipulated_parameters"]

    while not stop_event.is_set():
        try: 
            # Esegue un passo della simulazione.
            logger.info("Esecuzione di un nuovo passo della generazione del dataset.")

            #Genera uno scenario critico in cui deve operare il nodo selezionato
            scenario= generate_network_scenario(
                configuration["simulation_parameters"]["measured_parameters"],
                configuration["topology"]["network_description"],
                model_name=_scenario_generation_server
            )

            #Selezione un random goal tra i goal della rete e ne estrae la descrizione
            selected_goal = random.choice(configuration["topology"]["network_goals"])
            goal_desc = selected_goal["description"]
            


            insight= generate_insight(
                kind="malevolo",
                network_description=network_description, 
                data=scenario,
                manipulated_parameters=manipulated_parameters,
                goal_description="Congestionare la rete",
                model_name=_insight_generation_server
            )

            data= {
                "kind":"malevolo",
                "goal":"Congestionare la rete",
                "goal_id":selected_goal["id"],
                "scenario": scenario["scenario_description"],
                "measured_parameters":scenario["measured_parameters"],
                "manipulated_parameters":insight["manipulated_parameters"],
                "behavior":insight["behavior"],
                "confidence":insight["confidence"]
            }

            dataset.append(data)


            insight= generate_insight(
                kind="benevolo",
                network_description=network_description, 
                data=scenario,
                manipulated_parameters=manipulated_parameters,
                goal_description=goal_desc,
                model_name=_insight_generation_server
            )

            data= {
                "kind":"benevolo",
                "goal":goal_desc,
                "goal_id":selected_goal["id"],
                "scenario": scenario["scenario_description"],
                "measured_parameters":scenario["measured_parameters"],
                "manipulated_parameters":insight["manipulated_parameters"],
                "behavior":insight["behavior"],
                "confidence":insight["confidence"]
            }

            dataset.append(data)

        except Exception as exc:
                logger.warning(
                    f"Eccezione sollevata e ignorata: {exc}",
                    exc_info=True
                )
                continue

        save_dataset(dataset,PURE_DATASET)

    logger.info("Generazione dataset terminata.")
    logger.info(f"Numero di elementi nel dataset: {len(dataset)}")
    
