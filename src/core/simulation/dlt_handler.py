import numpy as np
import faiss
import logging
import time

logger = logging.getLogger(__name__)

_DLT=None
_INDEX=None

def init_dlt():
    """Inizializza il DLT 
    
    Args:
        None

    Returns:
        None
    """
    global _DLT

    logger.info(
        "Inizializzazione DLT ",
    )

    _DLT=[]
    
    return

def reset_dlt():
    """Azzera il DLT 
    
    Args:
        None

    Returns:
        None
    """
    global _DLT

    _DLT = None
    
    return

def is_dlt_setup():
    """Restituisce True se il DLT è inizializzato e non vuoto."""
    return bool(_DLT)

def _vectorize(param, param_configuration):
    """Normalizza e pesa i parametri di un record.
    
    Args:
        param (dict): dizionario contenente per ogni parametro specificato in param_configuration un valore
        param_configuration (list[dict]):configurazione utilizzata per la costruzione del vettore. 
                Ogni dizionario raccoglie le info relative ad un parametro, 
                deve contenere le chiavi "name", "weight"", "min","max"
    Notes: 
        weight deve essere sempre maggiore di zero
    
    Returns:
        np.ndarray: vettore normalizzato e pesato di tipo float32, contenente i valori in param, normalizzati e pesati 
        secondo i parametri di configurazione contenuti in param_configuration.
        I valori di param sono riportati nella lista secondo l'ordine in cui essi occorrono in param_configuration
    """

    total_weight = sum(
        config["weight"] for config in param_configuration
    )

    vector = []
    for config in param_configuration:
        name = config["name"]
        value = float(param[name])

        # Normalizza il valore nell'intervallo [0, 1].
        normalized = (
            (value - config["min"]) /
            (config["max"] - config["min"])
        )

        weight = config["weight"]
        weighted = normalized * math.sqrt(weight / total_weight)

        vector.append(weighted)

    # Restituisce il vettore come array NumPy float32,
    # formato richiesto da FAISS.
    return np.array(vector, dtype=np.float32)

def _add_index(param, param_configuration):
    """
    Aggiunge nuovi record all'indice FAISS basato sull'algoritmo HNSW
    (Hierarchical Navigable Small World).
    
    Se l'indice FAISS non è ancora stato inizializzato, lo crea
    utilizzando le dimensioni delle feature disponibili.

    Args:
        new_record (dict): il nuovo record da indicizzare.
            Verrà indicizzato esattamente tutto il contenuto del dizionario
        data_configuration(dict): configurazione utilizzata per la costruzione dei vettori. 
                Deve contenere le chiavi:
                - manipulated_parameters (list[dict]): dizionari aventi chiave "weight", "min","max"

    Returns:
        None
        
    """
    global _INDEX

    #costruisce l'array delle feature normalizzate
    X_norm = np.array([
        _vectorize(
            param,
            param_configuration,
        )
    ], dtype=np.float32)
    
    # Se l'indice non è ancora stato creato, inizializza FAISS.
    if _INDEX is None:

        _INDEX = faiss.IndexHNSWFlat(
            len(param), #numero di feature che compongono ogni vettore da indicizzare
            32, #numero massimo di collegamenti che ogni nodo può avere con altri nodi nel grafo
            faiss.METRIC_L2 #usa distanza euclidea al quadrato per calcolare la distanza tra vettori
        )

        # Configura i parametri di costruzione della struttura HNSW., valore suggerito 200
        _INDEX.hnsw.efConstruction = 100

        # Configura il numero di candidati esplorati durante la ricerca., valore suggerito 64
        _INDEX.hnsw.efSearch = 32

    # Aggiunge il singolo vettore normalizzato all'indice FAISS.
    _INDEX.add(X_norm)

    logger.info(
        "Record aggiunto all'indice FAISS. Totale record indicizzati: %d",
        _INDEX.ntotal
    )

def add_record(new_record, param_configuration):
    """Aggiunge un nuovo record al DLT
    
     Args:
        new_record (dict): Elemento da aggiungere a local_knowledge. Deve contenere le chiavi:
                -node_id: id del nodo che ha prodotto il record
                -goal_id: id del goal che il nodo ha seguito
                - context (dict): Informazioni relative all'azione,con chiavi scenario e measured_parameters.
                - action (dict): Informazioni relative al contesto, con chiavi manipulated_parameters, behavior e confidence.
        param_configuration(dict): configurazione utilizzata per la costruzione dei vettori. 
                Deve contenere per ogni measured parameter chiavi "weight", "min","max" 

    Returns:
        None: il DLT viene aggiornato direttamente aggiungendo il nuovo elemento.
    """
    
    global _DLT

    dlt.append(new_record)
    _add_index(new_record["context"]["measured_parameters"])

    index.add(vector)    

    
    logger.info(
        "Record aggiunto. Numero totale di elementi: %d",
        len(_DLT)
    )

def search_dlt(query, k, max_distance,param_configuration):
    """Restituisce i k elementi di search space più simili alla query.
    
    Args:
        query (list): elemento di riferimento per la ricerca
        k (int): Numero massimo di elementi da restituire.
        max_distance (float): Distanza massima ammessa tra la query di riferimento  e gli elementi candidati
        param_configuration(dict): configurazione utilizzata per la costruzione dei vettori. 
                Deve contenere per ogni measured parameter chiavi "weight", "min","max" 

    Notes:
        k deve essere un intero positivo,
        max_distance deve essere un valore numerico positivo

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
        "Avvio ricerca nel DLT. candidati: %d",
        _INDEX.ntotal if _INDEX is not None else 0
    )

    # Verifica che l'indice FAISS sia stato inizializzato
    # e contenga almeno un vettore.
    if _INDEX is None or _INDEX.ntotal == 0:

        elapsed_time = time.perf_counter() - start_time

        logger.warning(
            "Indice FAISS vuoto. Durata ricerca: %.6f secondi",
            elapsed_time
        )

        return []


    # Normalizza e pesa i parametri della query utilizzando
    # la stessa configurazione impiegata durante l'indicizzazione.
    # reshape(1, -1) trasforma il vettore in una matrice
    # bidimensionale, con una riga e tante colonne quante sono
    # le feature indicizzate.
    query_vector = _vectorize(
        query,
        param_configuration
    ).reshape(1, -1)

    # Esegue la ricerca FAISS dei k vettori più vicini alla query.
    # distances contiene le distanze L2 al quadrato.
    # indices contiene gli indici dei record corrispondenti nella DLT.
    distances, indices = _INDEX.search(query_vector, k)

    
    results = []
    for distance, idx in zip(distances[0], indices[0]):

        # FAISS restituisce -1 quando non trova un candidato valido.
        if idx < 0:
            continue

        # Converte la distanza L2 al quadrato restituita da FAISS
        # nella distanza euclidea tradizionale.
        euclidean_distance = float(np.sqrt(distance))

        if euclidean_distance <= max_distance:

            results.append({
                "index": int(idx),
                "distance": euclidean_distance
            })

    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "Ricerca nel DLT completata. "
        "Risultati: %d. Durata: %.6f secondi",
        len(results),
        elapsed_time
    )

    return results

