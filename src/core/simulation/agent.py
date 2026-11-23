from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ===========================================================================
# CONFIGURAZIONE DELLO SPAZIO DELLE AZIONI
# ===========================================================================

# SUMO usa normalmente questi codici per gli stati dei semafori:
# r = rosso, G = verde, y = giallo.
COLORS = ("r", "G", "y")

# Tutti i valori interi da 3 a 120 secondi, inclusi.
DURATIONS = tuple(range(3, 121))

# Ogni azione è una coppia (colore, durata).
# 3 colori * 118 durate = 354 azioni possibili per ciascun link.
ACTION_CANDIDATES: tuple[tuple[str, int], ...] = tuple(
    (color, duration)
    for color in COLORS
    for duration in DURATIONS
)
NUM_ACTIONS = len(ACTION_CANDIDATES)

def get_max_state_time():
    """restituisce la durata massima di uno stato nelle scelte possibili dell'agente"""
    return 120


# ===========================================================================
# STRUTTURA DELLE TRANSIZIONI E REPLAY BUFFER
# ===========================================================================

@dataclass
class Transition:
    """Una transizione completa dell'ambiente per tutti i link gestiti."""
    state: np.ndarray
    action_indices: tuple[int, ...]
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Memoria FIFO che conserva le transizioni dell'agente."""

    def __init__(self, capacity: int = 50_000, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity deve essere positivo.")
        self.memory: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(
        self,
        state: np.ndarray,
        action_indices: Sequence[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action_indices=tuple(int(x) for x in action_indices),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# ===========================================================================
# RETE Q AUTOREGRESSIVA
# ===========================================================================

class AutoregressiveQNetwork(nn.Module):
    """
    Encoder comune dello stato globale e decoder Q condiviso.

    Il decoder riceve:
      - le caratteristiche dello stato globale;
      - la posizione del link da decidere;
      - le azioni già selezionate per i link precedenti.

    Restituisce 354 valori Q: uno per ogni combinazione colore/durata.
    Il numero di link è configurabile e non è più fissato a quattro.
    """

    def __init__(
        self,
        observation_size: int,
        num_signals: int,
        num_actions: int = NUM_ACTIONS,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.observation_size = observation_size
        self.num_signals = num_signals
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # L'embedding identifica la posizione del link nell'elenco link_ids.
        self.position_embedding = nn.Embedding(num_signals, 16)

        # Un vettore one-hot per ogni link, contenente l'azione già scelta.
        prefix_size = num_signals * num_actions

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + 16 + prefix_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(
        self,
        states: torch.Tensor,
        position: int,
        prefix_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        states:         [batch, observation_size]
        prefix_indices: [batch, num_signals], -1 per link non ancora decisi
        output:         [batch, num_actions]
        """
        batch_size = states.shape[0]
        state_features = self.encoder(states)

        positions = torch.full(
            (batch_size,),
            int(position),
            dtype=torch.long,
            device=states.device,
        )
        position_features = self.position_embedding(positions)

        # Codifica one-hot delle azioni già selezionate.
        prefix_one_hot = torch.zeros(
            batch_size,
            self.num_signals,
            self.num_actions,
            dtype=states.dtype,
            device=states.device,
        )

        for previous_position in range(position):
            indices = prefix_indices[:, previous_position]
            valid = indices >= 0
            if valid.any():
                rows = torch.arange(batch_size, device=states.device)[valid]
                prefix_one_hot[
                    rows, previous_position, indices[valid]
                ] = 1.0

        prefix_features = prefix_one_hot.reshape(batch_size, -1)

        decoder_input = torch.cat(
            (state_features, position_features, prefix_features), dim=1
        )
        return self.decoder(decoder_input)


# ===========================================================================
# AGENTE DQN
# ===========================================================================

class DQNAgent:
    def __init__(
        self,
        link_ids: Sequence[str],
        observation_size: int = 42,
        hidden_size: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.10,
        epsilon_decay: float = 0.999,
        batch_size: int = 64,
        replay_capacity: int = 50_000,
        target_update_frequency: int = 500,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        # Gli ID definiscono quali link controlla l'agente e in quale ordine.
        # L'ordine deve rimanere invariato durante tutta la simulazione.
        self.link_ids = tuple(int(link_id) for link_id in link_ids)

        if not self.link_ids:
            raise ValueError("Devi fornire almeno un linkID.")
        if len(set(self.link_ids)) != len(self.link_ids):
            raise ValueError("I linkID devono essere univoci.")
        if observation_size <= 0:
            raise ValueError("observation_size deve essere positivo.")
        if hidden_size <= 0:
            raise ValueError("hidden_size deve essere positivo.")
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate deve essere positivo e finito.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma deve essere compreso tra 0 e 1.")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError("Valori epsilon non validi.")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay deve essere compreso tra 0 e 1.")
        if batch_size <= 0:
            raise ValueError("batch_size deve essere positivo.")
        if target_update_frequency <= 0:
            raise ValueError("target_update_frequency deve essere positivo.")

        # Numero di link determinato dalla simulazione
        self.num_signals = len(self.link_ids)

        self.rng = random.Random(seed)

        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # La rete riceve il numero effettivo di link.
        with torch.random.fork_rng(devices=[]):

            if seed is not None:
                torch.manual_seed(seed)

            self.policy_net = AutoregressiveQNetwork(
                observation_size,
                self.num_signals,
                NUM_ACTIONS,
                hidden_size,
            ).to(self.device)

            self.target_net = AutoregressiveQNetwork(
                observation_size,
                self.num_signals,
                NUM_ACTIONS,
                hidden_size,
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.replay_buffer = ReplayBuffer(
            replay_capacity,
            seed=seed,
        )
        self.learn_steps = 0

    # -----------------------------------------------------------------------
    # Conversione e validazione delle azioni
    # -----------------------------------------------------------------------


    @staticmethod
    def action_to_index(action: tuple[str, int]) -> int:
        """Converte una coppia (colore, durata) nell'indice dell'azione."""

        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError(
                "L'azione deve essere una coppia (colore, durata)."
            )

        color = action[0]
        duration = action[1]

        if not isinstance(color, str):
            raise ValueError("Il colore dell'azione deve essere una stringa.")

        if (
            isinstance(duration, (bool, np.bool_))
            or not isinstance(duration, (int, np.integer))
        ):
            raise ValueError("La durata dell'azione deve essere un intero.")

        try:
            return ACTION_CANDIDATES.index((color, int(duration)))
        except ValueError as exc:
            raise ValueError(
                f"Azione {action!r} non disponibile."
            ) from exc

    @staticmethod
    def index_to_action(index: int) -> tuple[str, int]:
        """Converte l'indice interno nella coppia (colore, durata)."""

        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
        ):
            raise ValueError("L'indice dell'azione deve essere un intero.")

        if not 0 <= index < NUM_ACTIONS:
            raise ValueError(f"Indice azione non valido: {index}")

        return ACTION_CANDIDATES[index]

    def _normalize_action(
        self,
        action: Mapping[str, tuple[str, int]] | Sequence[tuple[str, int]]
        | Sequence[int],
    ) -> tuple[int, ...]:
        """
        Accetta:
          - un dizionario {linkID: (colore, durata)};
          - una sequenza di coppie, nello stesso ordine di self.link_ids;
          - una sequenza di indici interni delle azioni.

        Il dizionario è la forma consigliata perché associa esplicitamente
        ogni decisione al relativo linkID.
        """
        if isinstance(action, Mapping):
            if set(action.keys()) != set(self.link_ids):
                raise ValueError(
                    "Le chiavi dell'azione devono coincidere esattamente "
                    "con i linkID configurati nell'agente."
                )
            return tuple(
                self.action_to_index(action[link_id])
                for link_id in self.link_ids
            )

        if len(action) != self.num_signals:
            raise ValueError(
                f"L'azione deve contenere {self.num_signals} elementi, "
                "uno per ogni linkID."
            )

        if all(
            isinstance(x, (int, np.integer))
            and not isinstance(x, (bool, np.bool_))
            for x in action
        ):
            indices = tuple(int(x) for x in action)

            if any(x < 0 or x >= NUM_ACTIONS for x in indices):
                raise ValueError("Indice azione fuori intervallo.")

            return indices

        return tuple(
            self.action_to_index((item[0], item[1]))
            for item in action
        )

       

    def _state_tensor(self, observation: Iterable[float]) -> torch.Tensor:
        """Controlla e converte lo stato in un tensore PyTorch."""
        state = np.asarray(observation, dtype=np.float32).reshape(-1)

        if state.size != self.observation_size:
            raise ValueError(
                f"Osservazione con {state.size} feature; "
                f"attese {self.observation_size}."
            )
        if not np.isfinite(state).all():
            raise ValueError("L'osservazione contiene NaN o valori infiniti.")

        return torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )

    # -----------------------------------------------------------------------
    # Scelta autoregressiva dell'azione
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _greedy_action_indices(
        self,
        state_batch: torch.Tensor,
        network: AutoregressiveQNetwork,
    ) -> tuple[int, ...]:
        """Sceglie in sequenza un'azione per ciascun link."""
        if state_batch.shape[0] != 1:
            raise ValueError("_greedy_action_indices si aspetta un solo stato.")

        prefix = torch.full(
            (1, self.num_signals),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        selected: list[int] = []

        for position in range(self.num_signals):
            q_values = network(state_batch, position, prefix)
            chosen = int(torch.argmax(q_values, dim=1).item())
            prefix[0, position] = chosen
            selected.append(chosen)

        return tuple(selected)

    def select_action(
        self,
        observation: Iterable[float],
        explore: bool = True,
    ) -> dict[str, tuple[str, int]]:
        """
        Restituisce un dizionario:
            {linkID: (codice_colore, durata)}

        Con explore=True applica epsilon-greedy;durante l'esplorazione
        campiona uniformemente una delle 354 azioni per ogni link.

        #     se explore=True, l'agente applica epsilon-greedy,
        #     quindi può esplorare nuove azioni oppure sfruttare
        #     quelle che ha imparato.
        #
        #     se explore=False, non esplora e sceglie le azioni
        #     migliori secondo la rete Q.

        Returns:
            azione: dizionario che ad ogni link_index_id associa una tupla colore durata

        """
        state = self._state_tensor(observation).unsqueeze(0)

        if explore and self.rng.random() < self.epsilon:
            indices = tuple(
                self.rng.randrange(NUM_ACTIONS)
                for _ in range(self.num_signals)
            )
        else:
            self.policy_net.eval()
            indices = self._greedy_action_indices(state, self.policy_net)
            self.policy_net.train()

        # L'associazione linkID -> azione evita di dipendere dalle direzioni.
        return {
            int(link_id): self.index_to_action(index)
            for link_id, index in zip(self.link_ids, indices)
        }

    # -----------------------------------------------------------------------
    # Memorizzazione delle esperienze
    # -----------------------------------------------------------------------

    def remember(
        self,
        state: Iterable[float],
        action: Mapping[str, tuple[str, int]] | Sequence[tuple[str, int]]
        | Sequence[int],
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ) -> None:
        """Salva una transizione nel replay buffer."""
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state_array = np.asarray(next_state, dtype=np.float32).reshape(-1)

        if not np.isfinite(state_array).all():
            raise ValueError("Lo stato contiene NaN o valori infiniti.")

        if not np.isfinite(next_state_array).all():
            raise ValueError("Il next_state contiene NaN o valori infiniti.")

        if not np.isfinite(reward):
            raise ValueError("La reward deve essere un valore finito.")

        if state_array.size != self.observation_size:
            raise ValueError("Dimensione dello stato non coerente.")
        if next_state_array.size != self.observation_size:
            raise ValueError("Dimensione del next_state non coerente.")

        indices = self._normalize_action(action)

        self.replay_buffer.add(
            state_array, indices, reward, next_state_array, done
        )

    # -----------------------------------------------------------------------
    # Calcolo del valore Q congiunto
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _max_joint_q(
        self,
        network: AutoregressiveQNetwork,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Calcola la Q congiunta massima con selezione autoregressiva."""
        batch_size = next_states.shape[0]
        prefix = torch.full(
            (batch_size, self.num_signals),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        total = torch.zeros(batch_size, device=self.device)

        for position in range(self.num_signals):
            q_values = network(next_states, position, prefix)
            best_values, best_indices = q_values.max(dim=1)

            total += best_values
            prefix[:, position] = best_indices

        return total

    @torch.no_grad()
    def _double_dqn_target(
        self,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Double DQN autoregressiva.

        La policy network seleziona le azioni.
        La target network valuta le azioni selezionate.
        """

        batch_size = next_states.shape[0]

        prefix = torch.full(
            (batch_size, self.num_signals),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        total_q = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=self.device,
        )

        for position in range(self.num_signals):

            # Policy network: selezione dell'azione
            policy_q = self.policy_net(
                next_states,
                position,
                prefix,
            )

            best_indices = policy_q.argmax(dim=1)

            # Target network: valutazione dell'azione
            target_q = self.target_net(
                next_states,
                position,
                prefix,
            )

            selected_q = target_q.gather(
                1,
                best_indices.unsqueeze(1),
            ).squeeze(1)

            total_q += selected_q

            # Aggiornamento del contesto autoregressivo
            prefix[:, position] = best_indices

        return total_q
    # -----------------------------------------------------------------------
    # Aggiornamento DQN
    # -----------------------------------------------------------------------

        
    def learn(self) -> float | None:
        """Aggiornamento Double DQN autoregressivo."""

        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.as_tensor(
            np.stack([t.state for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            np.asarray([t.action_indices for t in transitions]),
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            [t.reward for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.stack([t.next_state for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            [t.done for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        # Q delle azioni effettivamente eseguite
        current_q = self._joint_q_values(
            self.policy_net,
            states,
            actions,
        )

        # Target Double DQN
        with torch.no_grad():

            next_q = self._double_dqn_target(
                next_states
            )

            targets = (
                rewards
                + self.gamma * (1.0 - dones) * next_q
            )

        # Calcolo della loss
        loss = F.smooth_l1_loss(
            current_q,
            targets,
        )

        self.optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            10.0,
        )

        self.optimizer.step()

        self.learn_steps += 1

        # Aggiornamento rete target
        if self.learn_steps % self.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())


    def update_target_network(self) -> None:
        """Copia i pesi della rete policy nella rete target."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    # -----------------------------------------------------------------------
    # Salvataggio e caricamento
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Salva reti, ottimizzatore e configurazione dei link/azioni."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "observation_size": self.observation_size,
                "link_ids": self.link_ids,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
                "action_candidates": ACTION_CANDIDATES,
                "hidden_size": self.hidden_size,
            },
            destination,
        )
    
    def load(self, path: str | Path) -> None:
        """Carica un checkpoint solo se stato, link e azioni coincidono."""

        path = Path(path)

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        # Verifica dimensione dello stato
        if checkpoint["observation_size"] != self.observation_size:
            raise ValueError(
                "Il checkpoint usa una dimensione di stato diversa."
            )

        # Verifica dimensione dei layer nascosti
        if checkpoint["hidden_size"] != self.hidden_size:
            raise ValueError(
                "Il checkpoint usa una dimensione hidden_size diversa."
            )

        # Verifica dei link controllati
        saved_link_ids = checkpoint["link_ids"]

        if set(saved_link_ids) != set(self.link_ids):
            raise ValueError(
                "I linkID del checkpoint non coincidono con quelli "
                "configurati nell'agente."
            )

        # Verifica dello spazio delle azioni
        saved_actions = tuple(
            tuple(x) for x in checkpoint["action_candidates"]
        )

        if saved_actions != ACTION_CANDIDATES:
            raise ValueError(
                "Lo spazio delle azioni del checkpoint è diverso "
                "da quello configurato in questo agent.py."
            )

        # Caricamento delle reti neurali
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        # Caricamento dello stato dell'optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Ripristino dei parametri dell'agente
        self.epsilon = float(checkpoint["epsilon"])
        self.learn_steps = int(checkpoint["learn_steps"])

        print(f"Checkpoint caricato correttamente da: {path}")


    def _joint_q_values(
        self,
        network: AutoregressiveQNetwork,
        states: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcola la Q congiunta delle azioni realmente eseguite.
        """

        batch_size = states.shape[0]

        prefix = torch.full(
            (batch_size, self.num_signals),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        joint_q = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=self.device,
        )

        for position in range(self.num_signals):

            q_values = network(
                states,
                position,
                prefix,
            )

            selected = action_indices[:, position]

            selected_q = q_values.gather(
                1,
                selected.unsqueeze(1),
            ).squeeze(1)

            joint_q += selected_q

            prefix[:, position] = selected

        return joint_q


    @torch.no_grad()
    def debug_q_values(self, observation):

        state = self._state_tensor(
            observation
        ).unsqueeze(0)

        prefix = torch.full(
            (1, self.num_signals),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        self.policy_net.eval()

        for position in range(self.num_signals):

            q_values = self.policy_net(
                state,
                position,
                prefix,
            )

            values, indices = torch.topk(
                q_values,
                k=5,
                dim=1,
            )

            print(f"\nLink posizione {position}")

            for value, index in zip(
                values[0].tolist(),
                indices[0].tolist(),
            ):

                action = self.index_to_action(index)

                print(
                    f"Azione: {action} | Q-value: {value:.5f}"
                )

            best_action = indices[0, 0].item()

            prefix[0, position] = best_action

        self.policy_net.train()
    



