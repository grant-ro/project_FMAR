"""Modulo principale dell'applicazione.

Inizializza i componenti necessari e avvia
l'esecuzione dell'applicazione.
"""

import logging

from flask import Flask, render_template
from flask_socketio import SocketIO

import core.config_loader as config_loader
from core.operation_coordinator import (
    initialize,
    get_configuration_snapshot,
    request_reload,
    request_start_simulation,
    request_reset_simulation,
    request_stop_long_simulation
)

from core.config_watcher import start_config_watcher
from core.simulation.simulation_runner import run_long_simulation, run_short_simulation, run_dataset_generation

BASE_DIR = Path(__file__).resolve().parents[1
LOG_DIR = BASE_DIR / "data" / "output" / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)]

logging.basicConfig(
    level=logging.INFO,
    filename=str(LOG_DIR / "app.log"),
    format="%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s",
    encoding="utf-8")
logger = logging.getLogger(__name__)

app = Flask(__name__)

socketio = SocketIO(
    app, 
    async_mode="threading",
)


@app.route("/")
def index():
    """Gestisce la pagina principale dell'applicazione
    
    Carica il template HTML e lo restituisce al browser.

    Returns:
        Contenuto HTML renderizzato a partire dal template``index.html``.
    """
    return render_template("index.html")

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Errore HTTP 500: %s", error)
    return "Errore interno del server. Controlla il log del terminale.", 500

@socketio.on("connect")
def handle_connect(auth=None):
    """Gestisce la connessione di un client Socket.IO.
    
    Invia la configurazione attuale al client appena connesso.
    
    Args:
        auth: Dati di autenticazione del client, se forniti. Può essere ``None``
    
    Returns:
        None
    
    Notes: Emette un evento ``topology_updated`` contenente la topologia
        attuale della rete, per notificare ai client la configurazione aggiornata.
    """
    logger.info("Client Socket.IO connesso.")

    configuration = get_configuration_snapshot()

    socketio.emit(
        "topology_updated",
        configuration["topology"],
    )

@socketio.on("disconnect")
def handle_disconnect(reason=None):
    """Gestisce la sconnessione di un client Socket.IO.
    Args:
        reason: motivo della sconnessione
    
    Returns:
        None
    """
    
    logger.info("Client Socket.IO disconnesso.")

  

@socketio.on("short_simulation")
def handle_short_simulation(arg):
    """Gestisce la richiesta Socket.IO di avvio di una simulazione breve.

    Returns:
        dict: Dizionario contenente:
            - ok (bool): Indica se l'operazione è stata avviata.
            - message (str): Messaggio descrittivo dell'esito.
    """

    logger.info("Richiesta di avvio di una simulazione breve ricevuta.")

    ok, message = request_start_simulation(
        socketio=socketio,
        operation_kind="short",
        simulation_function=run_short_simulation,
        function_argument=arg,
        supports_stop=False
    )

    return {
        "ok": ok,
        "message": message,
    }

@socketio.on("reset_simulation")
def handle_reset_simulation():
    """Gestisce la richiesta Socket.IO di reset di una simulazione brevE

    Returns:
        dict: Dizionario contenente:
            - ok (bool): Indica se il reset è stato avviato.
            - message (str): Messaggio descrittivo dell'esito.
    
    """
    logger.info("Richiesta di reset dello stato della simulazione ricevuta.")

    ok, message = request_reset_simulation(
        socketio=socketio
    )

    return {
        "ok": ok,
        "message": message,
    }


@socketio.on("start_long_simulation")
def handle_start_long_simulation():
    """Gestisce la richiesta Socket.IO di avvio di una simulazione lunga.

    Returns:
        dict: Dizionario contenente:
            - ok (bool): Indica se l'operazione è stata avviata.
            - message (str): Messaggio descrittivo dell'esito.
    """

    logger.info("Richiesta di avvio di una simulazione lunga ricevuta.")

    ok, message = request_start_simulation(
        socketio=socketio,
        operation_kind="long_simulation",
        simulation_function=run_long_simulation,
        supports_stop=True,
    )

    return {
        "ok": ok,
        "message": message,
    }

@socketio.on("stop_long_simulation")
def handle_stop_long_simulation():
    """Gestisce la richiesta Socket.IO di stop di una simulazione lunga.

    Returns:
        dict: Dizionario contenente:
            - ok (bool): Indica se la richiesta di stop è stata inviata.
            - message (str): Messaggio descrittivo dell'esito.
    """

    logger.info("Richiesta di stop di una simulazione lunga ricevuta.")

    ok, message = request_stop_long_simulation(socketio, operation_kind = "long_simulation")

    return {
        "ok": ok,
        "message": message,
    }

@socketio.on("start_dataset_generation")
def handle_start_dataset_generation():
    """Gestisce la richiesta Socket.IO di avvio di generazione del dataset.

    Returns:
        dict: Dizionario contenente:
            - ok (bool): Indica se l'operazione è stata avviata.
            - message (str): Messaggio descrittivo dell'esito.
    """

    logger.info("Richiesta di avvio di generazione del dataset ricevuta.")

    ok, message = request_start_simulation(
        socketio=socketio,
        operation_kind="dataset_generation",
        simulation_function=run_dataset_generation,
        supports_stop=True,
    )

    return {
        "ok": ok,
        "message": message,
    }

@socketio.on("stop_dataset_generation")
def handle_stop_dataset_generation():
    """Gestisce la richiesta Socket.IO di stop della generazione del dataset.

    Returns:
        dict: Dizionario contenente:
            - ok (bool): Indica se la richiesta di stop è stata inviata.
            - message (str): Messaggio descrittivo dell'esito.
    """

    logger.info("Richiesta di stop della generazione del dataset ricevuta.")
    
    ok, message = request_stop_long_simulation(socketio, operation_kind = "dataset_generation")

    return {
        "ok": ok,
        "message": message,
    }


def main():
    """Avvia l'applicazione"""

    # Caricamento iniziale della configurazione dell'applicazione
    initial_configuration = config_loader.load_configuration()
    initialize(initial_configuration)

    #Avvio watcher sui file di configurazione
    watcher = start_config_watcher(
        lambda file_path: request_reload(file_path)
    )

    logger.info("Avvio del server su 127.0.0.1:5000.")
    try:
        socketio.run(
            app,
            host="127.0.0.1",
            port=5000,
            debug=False,
            use_reloader=False,
        )
    finally:
        watcher.stop()  
        watcher.join()


if __name__ == "__main__":
    main()