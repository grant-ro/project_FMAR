
import numpy as np
import math
import logging
import time

logger = logging.getLogger(__name__)

def init_local_knowledge(topology_nodes):
    """Inizializza la conoscenza locale a partire dalla topologia dei nodi.

    Args:
        topology_nodes (list[dict]): Lista di dizionari che descrivono i nodi della topologia. Ogni dizionario contiene:
            - node_id (str): Identificativo univoco del nodo.
            - node_goals (list[str]): Lista degli identificativi dei goal associati al nodo.

    Returns:
        list[dict]: Struttura dati local_knowledge, realizzata mediante una lista
        contenente un dizionario per ogni nodo della topologia. Ogni dizionario include:
            - nodo_id (str): Identificativo del nodo.
            - Una chiave per ogni goal associato al nodo, avente nome pari all'identificativo del goal,
             il cui valore è un dizionario contenente:
                    - data (list): Lista inizialmente vuota.
                    - context_vectors (list): Lista inizialmente vuota.
            
"""

    logger.info(
        "Inizializzazione conoscenza locale per %d nodi",
        len(topology_nodes)
    )

    local_knowledge = []

    for node in topology_nodes:
        node_knowledge = {
            "nodo_id": node["node_id"]
        }

        for goal in node.get("node_goals", []):
            node_knowledge[goal] = {
                "data": [],
                "context_vectors": [],
            }
    
        local_knowledge.append(node_knowledge)

    return local_knowledge


def _vectorize(param, param_configuration):
    """Normalizza e pesa i parametri di una conoscenza.
    
    Args:
        param (dict): dizionario contenente per ogni parametro specificato in param_configuration un valore
        param_configuration (list[dict]):configurazione utilizzata per la costruzione del vettore. 
                Ogni dizionario raccoglie le info relative ad un parametro, 
                deve contenere le chiavi "name", "weight"", "min","max"
    Notes: 
        weight deve essere sempre maggiore di zero
    
    Returns:
        list: lista contenente i valori in param, normalizzati e pesati 
        secondo i parametri di configurazione contenuti in param_configuration.
        I valori di param sono riportati nella lista secondo l'ordine in cui essi occorrono in param_configuration
    """

    vector = []

    total_weight = sum(config["weight"] for config in param_configuration)
    vector = []

    for config in param_configuration:
        name = config["name"]
        value = float(param[name])

        normalized = (value - config["min"]) / (
            config["max"] - config["min"]
        )

        weight = config["weight"]
        vector.append(normalized * math.sqrt(weight / total_weight))
    
    return vector

def add_knowledge(local_knowledge, new_data, data_configuration):
    """Aggiunge una nuova conoscenza alla conoscenza locale di un nodo per uno specifico goal
    
     Args:
        local_knowledge (dict): Struttura dati della conoscenza locale di un nodo per uno specifico goal, 
                in cui inserire il nuovo elemento di conoscenza.Deve contenere le chiavi:
                - context_vectors (list):  lista del context per ogni conoscenza locale
        new_data (dict): Elemento da aggiungere a local_knowledge. Deve contenere le chiavi:
                - action (dict): Informazioni relative all'azione.
                - context (dict): Informazioni relative al contesto.
        data_configuration(dict): configurazione utilizzata per la costruzione dei vettori. 
                Deve contenere le chiavi "weight", "min","max" per ogni measured_parameter 

    Returns:
        None: La struttura local_knowledge viene aggiornata direttamente,
            aggiungendo il nuovo elemento e i vettori calcolati a partire
            dai campi action e context.
    """
    
    local_knowledge["data"].append(new_data)
    context_vector = _vectorize(new_data["context"], data_configuration)
    local_knowledge["context_vectors"].append(context_vector)

    logger.info(
        "Conoscenza aggiunta. Numero totale di elementi: %d",
        len(local_knowledge["data"])
    )


def search_lk(query, k, max_distance, search_space, param_configuration):
    """Restituisce i k elementi di search space più simili alla query.
    
    Args:
        query (dict): elemento di riferimento per la ricerca, deve contenere una chiave measured_parameters
        k (int): Numero massimo di elementi da restituire.
        max_distance (float): Distanza massima ammessa tra la query di riferimento  e gli elementi candidati
        search_space(list): insieme dei possibili candidati
        param_configuration(dict): configurazione utilizzata per la costruzione dei vettori. 
                Deve contenere per ogni measured parameter chiavi "weight", "min","max"

    Notes:
        k deve essere un intero positivo,
        max_distance deve essere un valore numerico positivo
        query deve avere la medesima struttura interna degli elementi di search_space

    Returns: 
        Returns:
        list[dict]: Lista contenente al massimo k risultati, ordinati
            per distanza crescente dalla query. Ogni risultato contiene:
                - index (int): Indice del vettore in search_space.
                - distance (float): Distanza euclidea dalla query.
            Sono inclusi soltanto gli elementi la cui distanza non supera max_distance
    
    """
    start_time = time.perf_counter()

    logger.info(
        "Avvio ricerca nella conoscenza locale. candidati: %d",
        len(search_space)
        )

    #se la collezione è vuota la ricerca fallisce
    if not search_space["context_vectors"]:
        elapsed_time = time.perf_counter() - start_time
        logger.warning(
            "Spazio di ricerca vuoto. Durata ricerca: %.6f secondi",
            elapsed_time
        )
        return []

    query_vector=_vectorize(query["measured_parameters"], param_configuration)
    query_vector = np.asarray(query_vector, dtype=float)
    X = np.asarray(search_space, dtype=float)

    #Calcola la distanza euclidea tra ogni elemento della collezione e query
    distances = np.linalg.norm(X - query_vector, axis=1)

    #Nota: Crea un vettore di indici che permette successivamente di conservare
    #il collegamento tra ogni distanza calcolata e il vettore corrispondente nella matrice originale.
    indices = np.arange(len(search_space))

    #Filtra per distanza massima
    mask = distances <= max_distance #Produce una maschera booleana
    indices = indices[mask]         #applica la maschera booleana a indices
    distances = distances[mask]     #applica la maschera booleana a distances

    # Seleziona i k risultati più vicini
    order = np.argsort(distances)[:k]

    #Nota:le conversioni int() e float() servono a restituire normali tipi numerici Python,
    # anziché tipi scalari specifici di NumPy.
    results = [
        {
            "index": int(indices[j]),
            "distance": float(distances[j])
        }
        for j in order
    ]

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Ricerca nella conoscenza locale completata: Durata: %.6f secondi",
        elapsed_time
    ) 
    
    return results
   


