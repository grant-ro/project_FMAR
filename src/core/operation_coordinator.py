import copy
import logging
import threading
from collections import deque 
import core.config_loader as config_loader
from core.simulation.simulation_runner import (
    run_short_simulation,
    run_long_simulation,
    reset_simulation
)

"""
Coordinatore dell'operazioni di simulazione e di reloading dei file di configurazione:
- Accumula le richieste di reload in una coda.
- rinvia l'esecuzione di reload della configurazione al termine di op di simulazione in esecuzione al momento della richiesta di reload
- rifiuta la richiesta di esecuzione di un'operazione di simulazione se un'altra è già in corso
- mantiene la configurazione attuale dell'applicazione
"""

logger = logging.getLogger(__name__)

_state_lock = threading.Lock()

_reload_queue = deque() # coda dei file_path dei file di configurazione da ricaricare
_reloading = False  #stato di attivazione del ciclo di reload 

_active_configuration = None #configurazione attualmente usata dall'applicazione

_operation_active = False #stato di attivazione di una qualunque operazione di simulazione
_operation_kind = None #tipo di operazione di simulazione in esecuzione. Può assumere i seguenti valori "short_simulation" "long_simulation" "dataset_generation" "reset"
_stop_event_long_simulation = None
_stop_event_dataset_generation = None




def initialize(configuration):
    """Inizializza la configurazione usata dall'applicazione."""
    global _active_configuration

    with _state_lock:
        _active_configuration = copy.deepcopy(configuration)


def get_configuration_snapshot():
    """Restituisce una copia della configurazione attiva."""
    with _state_lock:
        if _active_configuration is None:
            raise RuntimeError("Configurazione non inizializzata.")

        return copy.deepcopy(_active_configuration)

def request_reload(file_path):
    """Registra una richiesta di reload per il file specificato in file_path in reload queue e prova ad applicarla.
    
    Args: 
        file_path: il path del file di cui si richiede il reload

    Returns: None
    """

    with _state_lock:
        _reload_queue.append(file_path)

    _apply_pending_reload()


def _apply_pending_reload():
    """Elabora in ordine le richieste di reload in reload queue 
        Una richiesta è elaborata solo se il coordinatore è libero (non ci sono altre operazioni attive).

    Returns:
        None

    Raises:
        OSError: Se il file di cui prova a fare il reload non può essere aperto o letto.
        json.JSONDecodeError: Se il file di cui prova a fare il reload non contiene un JSON valido.
        ValueError: Se il file di cui prova a fare il reload ha un formato diverso da quello atteso o non rispetta i prerequesiti definiti
    
    Notes:
        Un'eventuale eccezione è registrata sul file di log prima di essere propagata al chiamante
    
    """

    global _reloading

    while True:
        with _state_lock:
            if (
                not _reload_queue   #se la queue è vuota, il ciclo di reloading ha esaurito le richieste da servire e termina
                or _reloading   # se il ciclo di reloading è gia attivo da una richiesta precedente, la richiesta corrente di reload termina
                or _operation_active    #il ciclo di reload non si può attivare se un'altra operazione è già in esecuzione:
                                        #la configurazione non può cambiare mentre è in corso un'op di simulazione
                
            ):
                return

            file_path = _reload_queue.popleft()
            _reloading = True

        try:
            key, new_value = config_loader.load_config_file(file_path)

            with _state_lock:
                if key not in _active_configuration:
                    raise KeyError(
                        f"Chiave di configurazione non valida: {key}"
                )

                _active_configuration[key] = copy.deepcopy(new_value)


            reset_simulation() #resetta lo stato della simulazione finora effettuata con le short simulations

            logger.info(
                "Configurazione ricaricata correttamente dal file %s.",
                file_path,
            )
        finally:
            with _state_lock:
                _reloading = False


def _finish_operation(socketio, result=None, error=None):
    """Marca la fine effettiva dell'operazione e sblocca i reload.
    
    Args:
        socketio: Istanza Socket.IO utilizzata per inviare eventi al client.
        result: Risultato dell'operazione da comunicare al client in caso di completamento corretto dell'operazione. 
            Se None o vuoto, viene inviato un dizionario vuoto.
        error: Eccezione o errore verificatosi durante l'esecuzione dell'operazione.
            Se diverso da None, viene inviato il relativo messaggio al client.

    Returns:
        None
 
    """
    global _operation_active
    global _operation_kind
    global _stop_event

    with _state_lock:
        logger.info(
            "Simulazione %s terminata.",
            _operation_kind,
        )

        #riporta il coordinatore a uno stato coerente alla terminazione dell'operazione
        _operation_active = False 
        _operation_kind = None
        _stop_event = None

    if error is not None:
        #segnala al client l'avvenimento di un errore durante l'esescuzione dell'operazione
        socketio.emit(
            "operation_error",
            {"message": str(error)},
        )

        logger.warning(
            "Simulazione terminata con errore."
        )

    else:
        #comunica al client la terminazione con successo dell'operazione
        socketio.emit("operation_finished", result or {})

        

    _apply_pending_reload() #esegue, se ci sono, richieste di reload effettuate durante l'esecuzione dell'operazione 


def request_start_simulation(
        socketio,
        operation_kind,
        simulation_function,
        function_argument=None,
        supports_stop=False,
    ):
    """ Avvia una simulazione in background se il coordinatore è libero, 
    altrimenti la richiesta di simulazione fallisce.
    
    Args:
        socketio:
            Istanza Socket.IO utilizzata per inviare notifiche
            ai client e avviare un'attività in background.

        operation_kind (str):
            Identificativo del tipo di simulazione: "short" per una simulazione breve o "long" per una simulazione lunga.

        simulation_function (callable):
            Funzione che esegue la simulazione:
            Callable[[dict], Any] Se supports_stop è False
            Callable[[dict, threading.Event], Any] Se supports_stop è True

        function_argumemt: 
            argomento da passare alla funzione da invocare. Nota: viene usato solo per la simulazione di tipo short.

        supports_stop (bool):
            Indica se la simulazione supporta una richiesta di arresto tramite threading.Event. 

    Returns:
        tuple[bool, str]: Una tupla contenente:
            - True e un messaggio descrittivo se l'operazione è stata avviata correttamente.
            - False e un messaggio di errore se il server è occupato oppure la configurazione non è inizializzata.

    """

    global _operation_active
    global _operation_kind
    global _stop_event

    with _state_lock:
        if (
            _operation_active #se un'altra operazione di simulazione è in esecuzione, la richiesta di avvio della simulazione fallisce
            or _reloading   #se un ciclo di reloading è in esecuzione, la richiesta di avvio della simulazione fallisce
        ):
            return False, "Il server è occupato."

        if _active_configuration is None:   #se la configurazione non è stata ancora caricata, la simulazione non può essere eseguita e la richiesta di avvio fallisce
            return False, "Configurazione non inizializzata."

        configuration = copy.deepcopy(_active_configuration)

        # Crea l'evento solo per le operazioni interrompibili.
        if supports_stop:
            _stop_event = threading.Event()
        else:
            _stop_event = None

        _operation_active = True
        _operation_kind = operation_kind

    def worker():
        try:
            #invoca la corretta funzione che esegue la simulazione a seconda del tipo di simulazione richiesta
            if supports_stop:
                result = simulation_function(
                    configuration, _stop_event
                )
            else:
                result = simulation_function(configuration, function_argument)

        except Exception as exc:
            logger.exception(
                "Errore nell'esecuzione della simulazione %s.",
                operation_kind,
            )
            _finish_operation(socketio, error=exc)

        else:
            _finish_operation(socketio, result=result)

    logger.info(
        "Simulazione %s avviata.",
        operation_kind,
    )

    socketio.start_background_task(worker)

    return True, f"Operazione {operation_kind} avviata."

def request_reset_simulation(socketio):
    """Richiede il reset dello stato della simulazione.
    
    socketio:
            Istanza Socket.IO utilizzata per inviare notifiche
            ai client e avviare un'attività in background.
    
    Returns:
        tuple[bool, str]: Una tupla contenente:
            - True e un messaggio descrittivo se la richiesta di reset è stata inviata correttamente.
            - False e un messaggio di errore se il server è occupato oppure la configurazione non è inizializzata.

    """

    global _operation_active
    global _operation_kind

    with _state_lock:
        if (
            _operation_active #se un'altra operazione di simulazione è in esecuzione, la richiesta di avvio della simulazione fallisce
            or _reloading   #se un ciclo di reloading è in esecuzione, la richiesta di reset della simulazione fallisce
        ):
            return False, "Il server è occupato."

        if _active_configuration is None:
            return False, "Configurazione non inizializzata."

        _operation_active = True
        _operation_kind = "reset"

    def worker():
        try:
            reset_simulation()

        except Exception as exc:
            logger.exception("Errore durante il reset della simulazione.")
            _finish_operation(socketio, error=exc)

        else:
            _finish_operation(
                socketio,
                result={"message": "Simulazione resettata correttamente."},
            )

    logger.info("Reset della simulazione avviato.")
    socketio.start_background_task(worker)

    return True, "Reset della simulazione avviato."

def request_stop_long_simulation(socketio, operation_kind):
    """Richiede lo stop, senza dichiarare prematuramente la fine.
    
    Args:
        socketio:
            Istanza Socket.IO utilizzata per inviare notifiche
            ai client e avviare un'attività in background.
        operation_kind: stringa che specifica il tipo di operazione lunga da interrompere

    Returns:
        tuple[bool, str]: Una tupla contenente:
            - True e un messaggio descrittivo se la richiesta di stop è stata inviata correttamente.
            - False e un messaggio di errore se il server è occupato oppure la configurazione non è inizializzata.

    """

    with _state_lock:
        if (
            not _operation_active       #Se non c'è nessuna operazione attiva correntemente, la richiesta di stop operazione fallisce
            or operation_kind != _operation_kind    #se vi è un'op correntemente attiva, ma questa è  diversa da quella di cui si è richiesta l'interruzione, la richiesta di stop op fallisce
        ):
            return False, f"Non c'è un'operazione di tipo {operation_kind} attiva."

        
        stop_event = _stop_event

    stop_event.set()

    socketio.emit(
        "operation_stop_requested",
        {"operation": f"{operation_kind}"},
    )

    return True, "Richiesta di stop inviata."