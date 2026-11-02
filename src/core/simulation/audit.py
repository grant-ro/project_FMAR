from core.simulation.slm_handler import SLM_CONFIG, generate_insight

_nodes_nature= None
"""Dizionario annidato che associa a ogni nodo della topologia, identificato
dal proprio node_id, un dizionario contenente le seguenti informazioni:
    - 'true' (str): rappresenta la vera natura del nodo. 
    - 'known' (str): indica se la natura del nodo  nota al sistema.
    - 'score' (float): rappresenta il punteggio di affidabilità associato al nodo.
"""
_threshold= None
""" soglia dello score di ogni nodo, al di sotto della quale il nodo viene considerato malevolo"""

_damage_score= None
""" quantità di cui viene dimunuito lo score di un nodo di fronte a un comportamento sospetto"""

_merit_score= None
""" quantità di cui viene aumentato lo score di un nodo malevolo di fronte a un comportamento benevolo"""

def init_audit(audit_parameters, nodes_true_nature):
    """Inizializza l'audit
    
    Args: 
        audit_parameters: un dizionario con le seguenti chiavi:
            -damage_score: intero positivo compreso tra 0 e 100
            -threshold: intero postivo compreso tra 0 e 100
            -merit_score: intero positivo compreso tra 0 e 100
        nodes_true_nature:nodes_true_nature= Dizionario che associa per ogni nodo al suo node_id la stringa che rappresenta la sua reale natura (benevolo o malevolo)
    
    Returna:
            None
    """
    global _threshold, _damage_score, _merit_score

    _threshold = audit_parameters["threshold"]
    _damage_score = audit_parameters["damage_score"]
    _merit_score = audit_parameters["merit_score"]

    _init_nodes_nature(nodes_true_nature)



def _init_nodes_nature(nodes_true_nature):
    """
    Inizializza il dizionario nodes_nature per ogni nodo della topologia.

    Per ogni node_id è associato un dizionario così composto:
      - true: la vera natura del nodo (True o False), letta da nodes_true_nature.
      - known: True, perché la natura è inizialmente considerata nota.
      - score: 100, valore iniziale.
    """
    global _nodes_nature 
    _nodes_nature = {}

    for node_id, true_nature in nodes_true_nature.items():
        _nodes_nature[node_id] = {
            "true": true_nature,
            "known": "benevolo",
            "score": 100
        }

def get_node_known_nature( node_id):
    """Restituisce il valore 'known' del nodo.
    
    Args:
        node_id : Identificativo del nodo di cui recuperare lo stato di conoscenza della natura.
    
    Returns:
        str: "benevolo" o "malevolo" a seconda di quale sia la nota natura del nodo in quel momento 
    """
    return _nodes_nature[node_id]["known"]

def get_node_true_nature( node_id):
    """Restituisce il valore 'true' del nodo.

    Args: 
        node_id : Identificativo del nodo di cui recuperare lo stato di conoscenza della natura.
    
    Returns:
        str: "benevolo" o "malevolo" a seconda di quale sia la reale natura del nodo
    """
    return _nodes_nature[node_id]["true"]

def audit_reasoning(network_description,scenario, goal_description,manipulated_parameters, model_name="qwen-rete"):
    """Genera l'insight  di un nodo della rete affidandosi a un SLM.

    Args:
        network_description: Descrizione della natura della rete
        scenario: Dizionario con scenario_description e measured_parameters.
        manipulated_parameters: Lista di dizionari contenenti i parametri modificabili, ciascuno con nome, descrizione, unità di misura, min e max.
        goal_description: Obiettivo del nodo. Deve essere specificato per i nodi benevoli e deve essere None per quelli malevoli.
        model_name: nome del modello

    Returns:
        Dizionario JSON contenente:
            - behavior: descrizione del comportamento del nodo.
            - manipulated_parameters: dizionario con i valori proposti per ciascun parametro manipolabile 
            -confidence:il livello di fiducia del nodo nella soluzione proposta (valore tra 0 e 100).
    """

    insight = generate_insight(
       kind="benevolo",
        network_description=network_description,
        data=scenario,
        manipulated_parameters=manipulated_parameters,
        goal_description=goal_description,
        model_name=model_name
    )

    return insight

def _update_node_nature(node_id, score_delta):
    """ Aggiorna lo score di un nodo e verifica se la sua natura nota
    deve essere modificata in base alla soglia di affidabilità.

    Args:
        node_id: Identificativo del nodo da aggiornare.
        score_delta: Variazione da applicare allo score. Può essere positiva o negativa.
        
    Returns: 
        None
    """
    node_nature = _nodes_nature.get(node_id)
    node_nature["score"] += score_delta

    if node_nature["score"] < _threshold:
        node_nature["known"] = "malevolo"
    else: node_nature["known"] = "benevolo"



def audit_trust_evaluation (results, majority, node_id):
    """
    Aggiorna lo stato di fiducia dei nodi coinvolti nella generazione e
    nella votazione dell'insight, sulla base del risultato della votazione.

    Args: 
        results dict[str, bool]: dizionario associa a ogni identificativo di nodo (node_id) il voto espresso ("True" o "False").
        majority (bool): Esito della maggioranza dei votanti (True o False)

        node_id (str)) Identificativo del nodo che ha prodotto l'insight sottoposta a votazione.
            
    Notes:
        La funzione aggiorna la nota natura dei nodi coinvolti in base al loro ruolo nella 
            generazione e nella votazione dell'insight
        majority deve essere diversa True o False
        results deve essere una lista non vuota

            
    Returns: None


    """

    #Seleziona tutti i nodi che hanno espresso un voto diverso dalla maggioranza
    dissenting_nodes = [
        current_node_id
        for current_node_id, vote in results.items()
        if vote != majority
    ]

    #Diminuisce di un valore pari a _damage_score lo score di ogni nodo selezionato
    for current_node_id in dissenting_nodes:
        update_node_nature(current_node_id, -_damage_score)

    #Aumenta di un valore pari a _merit_Score lo score del nodo produttore dell'insight
    #se attualmente esso è classificato come malevolo
    producer = _nodes_nature.get(node_id)

    if producer["known"] == "malevolo":
        _update_node_nature(node_id, _merit_score)
        