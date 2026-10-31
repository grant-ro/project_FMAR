from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from core.config_loader import CONFIG_DIR, CONFIG_FILES

def start_config_watcher(on_change):
    """
    Crea un watcher che monitora i file in CONFIG_FILES

    Args:
        on_change: Callable[[str], None] - la funzione richiamata quando il watcher rileva un evento 
        di modifica, creazione o sostituzione su uno dei file in CONFIG_FILE.

    Returns:
        Observer: L'osservatore avviato, che può essere utilizzato per
        gestire il monitoraggio e arrestarlo quando necessario.
    """

    def handle_event(event):
        #ignora gli eventi sulla cartella
        if event.is_directory:
            return

        paths = [event.src_path]

        #recupera dest_path se esiste, per gestire correttamente l'evento di rinomina/spostamento 
        destination = getattr(event, "dest_path", None)
        if destination:
            paths.append(destination)

        for path in paths:
            if Path(path).name in CONFIG_FILES.values():
                on_change(path)

    event_handler = FileSystemEventHandler()

    event_handler.on_modified = handle_event
    event_handler.on_created = handle_event
    event_handler.on_moved = handle_event

    #Crea e configura un observer che monitori la cartella CONFIG_DIR,
    #rilevi gli eventi relativi a tale cartella e invochi il metodo corrispondente all'evento rilevato
    observer = Observer()
    observer.schedule(
        event_handler,
        str(CONFIG_DIR),
        recursive=False,
    )

    observer.start()

    return observer