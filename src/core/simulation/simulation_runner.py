
import threading
import requests
import random
from core.simulation.slm_handler import SLM_CONFIG, generate_network_scenario, generate_insight, evaluate_insight
from core.simulation.local_knowledge_handler import init_local_knowledge, vectorize, search, add_knowledge
from core.simulation.audit import init_audit, get_node_known_nature, get_node_true_nature, audit_reasoning, audit_trust_evaluation
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import time
import logging

logger = logging.getLogger(__name__)

_local_knowledge = None
"""Lista che associa un dizionario contenente la conoscenza locale del nodo per ogni nodo della topologia."""

_adjacency = None
"""Dizionario che associa per ogni nodo al suo node_id la lista dei node_id dei nodi adiacenti ad esso."""

_audit_state=False
"""Bool che rappresenta lo stato dell'audit e della sua conoscenza: True: configurato, False: non configurato"""

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


def _setup_simulation(configuration, nodes_true_nature):
    """Inizializza la conoscenza locale, la struttura di adiacenza della rete, 
        la conoscenza relativa alla natura dei nodi dell'audit. 
    
    Args:
        configuration: Dizionario contenente la configurazione della simulazione, inclusa la topologia della rete.
        nodes_true_nature= Dizionario che associa per ogni nodo al suo node_id la stringa che rappresenta la sua reale natura (benevolo o malevolo)

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
    _audit_state=True

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

    logger.info("Stato interno della simulazione resettato.")

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

    vectorized_scenario=vectorize(scenario["measured_parameters"],simulation_parameters["measured_parameters"])
    results= search(query=vectorized_scenario, k=k, max_distance=max_distance, search_space=local_knowledge["context_vectors"])    
    if results:
        flag=False
        #Calcola l'insight a partire dalle action effettuate nelle situazioni simili in passato
        if results[0]["distance"] == 0:
            index = results[0]["index"]
            insight= local_knowledge["data"][index]["action"].copy()
            
            logger.info(
                "Trovata una corrispondenza esatta nella conoscenza locale."
            )

        else:
            weighted_values = {}
            total_weight = 0.0

            for result in results:
                index = result["index"]
                distance = result["distance"]

                action = local_knowledge["data"][index]["action"]
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
                "Insight calcolata tramite media pesata di %d casi simili.",
            len(results)
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
            goal_description=goal_description
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
        goal_description=None
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
            logger.info(
                "Nodo %s:   INSIGHT RIFIUTATA, parametro '%s' fuori dai limiti consentiti "
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

        vectorized_scenario=vectorize(scenario["measured_parameters"],simulation_parameters["measured_parameters"])
        results= search(query=vectorized_scenario, k=k, max_distance=max_distance, search_space=goal_specific_knowledge["context_vectors"]) 
        if results:
            #Calcola l'insight a partire dalle action effettuate nelle situazioni simili in passato
            if results[0]["distance"] == 0:
                index = results[0]["index"]
                comparison_insight= goal_specific_knowledge["data"][index]["action"].copy()
                
                logger.info(
                    "Nodo %s: INSIGHT APPROVATA: trovata una corrispondenza esatta nella conoscenza locale.",
                    node_id
                )
        
            else:
                weighted_values = {}
                total_weight = 0.0

                for result in results:
                    index = result["index"]
                    distance = result["distance"]

                    action = goal_specific_knowledge["data"][index]["action"]
                    weight = 1.0 / distance 
                    total_weight += weight

                    for param, value in action.items():
                        weighted_values[param] = (
                            weighted_values.get(param, 0.0) + weight * value
                        )
                comparison_insight = {
                    param: value / total_weight
                    for param, value in weighted_values.items()
                }

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
                logger.info(
                    "Nodo %s: INSIGHT APPROVATA tramite confronto con la conoscenza locale.",
                    node_id
                )
                return True

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
        model_name="llama-3.2-1b",
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
            
        Any: Risultato della simulazione.
    """
    global _local_knowledge, _adjacency, _audit_state

    if _local_knowledge is None or _adjacency is None or _audit_state == False:
        _setup_simulation(configuration, data["nodes_true_nature"])
    
    logger.info(
        "Avvio simulazione breve: nodo=%s, natura=%s.",
        data["node_id"], data["node_nature"]
    )

    #Genera uno scenario critico in cui deve operare il nodo selezionato
    scenario= generate_network_scenario(
        configuration["simulation_parameters"]["measured_parameters"],
        configuration["topology"]["network_description"]
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

    if data["node_nature"] == "benevolo":        


        #Estrae la conoscenza locale associata al nodo e al goal selezionati
        for node in _local_knowledge:
            if node["nodo_id"] == data["node_id"]:
                goal_specific_knowledge=node[selected_goal]
                break

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
                    model_name=SLM_CONFIG["llama-3.1-8b"]["model_name"]
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
                    manipulated_parameters=configuration["simulation_parameters"]["manipulated_parameters"]
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
                new_data = {
                    "action": insight["insight"]["manipulated_parameters"],
                    "context":scenario["measured_parameters"]
                }
                logger.info(
                    "Nodo %s: maggioranza positiva; aggiornamento della conoscenza locale "
                    "per il goal %s.",
                    data["node_id"], selected_goal
                )

                add_knowledge(local_knowledge=goal_specific_knowledge, new_data=new_data, data_configuration=configuration["simulation_parameters"])
                
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

    logger.info("Avvio simulazione lunga.")

    # Inserire qui la logica della simulazione lunga.

    result = None

    while not stop_event.is_set():
        # Esegue un passo della simulazione.
        logger.info("Esecuzione di un nuovo passo della simulazione lunga.")
        # Aggiornare eventualmente result.

        # Esempio:
        # result = ...

        # Interrompere il ciclo quando la simulazione è terminata.
        break

    logger.info("Simulazione lunga terminata.")
    return result
