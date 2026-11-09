%%writefile /content/llama-project/scripts/train.py
import os
import json
import random
import gc
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

PROJECT_DIR = Path("/content/llama-project")

DATASET_FILE = (
    PROJECT_DIR
    / "data"
    / "input"
    / "training"
    / "llama_training_dataset.json"
)

OUTPUT_DIR = "/content/llama32-multitask-adapter-v2"
MERGED_DIR = "/content/llama32-multitask-merged-v2"

MAX_LENGTH = 2048
TEST_SIZE = 0.10
SEED = 42

NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

DO_MERGE = True
RESUME_FROM_CHECKPOINT = False


# ============================================================
# 2. CONTROLLI INIZIALI
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU non disponibile. In Colab seleziona Runtime > Change runtime type > T4 GPU."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Dataset non trovato: {DATASET_FILE}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nConfigurazione iniziale verificata.")


# ============================================================
# 3. CARICAMENTO TOKENIZER
# ============================================================

print("\nCaricamento tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Tokenizer caricato.")


# ============================================================
# 4. MODIFICA CHAT TEMPLATE
# ============================================================

print("\nConfigurazione chat template...")

template = tokenizer.chat_template

if not template:
    raise RuntimeError("Il tokenizer non contiene un chat template.")

if "{% generation" not in template and "{%- generation" not in template:

    old = """{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}"""

    new = """{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}
        {%- if message['role'] == 'assistant' %}
            {%- generation %}
                {{- message['content'] | trim }}
            {%- endgeneration %}
        {%- else %}
            {{- message['content'] | trim }}
        {%- endif %}
        {{- '<|eot_id|>' }}"""

    if old not in template:
        raise RuntimeError(
            "Non trovo il blocco originale del chat template. "
            "Interrompo il programma per evitare una modifica errata."
        )

    tokenizer.chat_template = template.replace(old, new)

if "{% generation" not in tokenizer.chat_template and "{%- generation" not in tokenizer.chat_template:
    raise RuntimeError("Il blocco generation non è stato aggiunto.")

print("Chat template aggiornato correttamente.")


# ============================================================
# 5. TEST DELLA MASCHERA SU UNA CONVERSAZIONE
# ============================================================

print("\nTest assistant mask...")

test_messages = [
    {
        "role": "system",
        "content": "Sei un assistente specializzato in simulazioni di traffico."
    },
    {
        "role": "user",
        "content": "Genera un nodo benevolo."
    },
    {
        "role": "assistant",
        "content": '{"type": "benevolent", "speed": 30}'
    },
]

test_result = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
)

test_ids = test_result["input_ids"]
test_mask = test_result["assistant_masks"]

if len(test_ids) != len(test_mask):
    raise RuntimeError("Lunghezza della maschera di test non valida.")

if sum(test_mask) == 0:
    raise RuntimeError("Maschera assistant vuota.")

test_target = [
    token_id
    for token_id, mask in zip(test_ids, test_mask)
    if mask == 1
]

print("Token totali:", len(test_ids))
print("Token assistente:", sum(test_mask))
print(
    "Risposta identificata:",
    tokenizer.decode(
        test_target,
        clean_up_tokenization_spaces=False
    )
)

print("Test della maschera superato.")


# ============================================================
# 6. CARICAMENTO DATASET JSON / JSONL
# ============================================================

print("\nCaricamento dataset...")

def load_dataset_file(path):
    path = Path(path)

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON non valido alla riga {line_number}: {exc}"
                    )
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return data["data"]
            if isinstance(data.get("messages"), list):
                return [data]

        raise ValueError(
            "Il JSON deve contenere una lista di conversazioni "
            "oppure una chiave 'data' con una lista."
        )

    raise ValueError("Il dataset deve essere un file .json o .jsonl.")


records = load_dataset_file(DATASET_FILE)

if len(records) < 2:
    raise ValueError(
        "Servono almeno due conversazioni per creare train e validation."
    )

print("Conversazioni caricate:", len(records))


# ============================================================
# 7. VALIDAZIONE DELLE CONVERSAZIONI
# ============================================================

def validate_messages(record, index):
    if not isinstance(record, dict):
        raise ValueError(
            f"Esempio {index}: ogni record deve essere un oggetto JSON."
        )

    messages = record.get("messages")

    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"Esempio {index}: manca la lista 'messages'."
        )

    for msg_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Esempio {index}, messaggio {msg_index}: formato non valido."
            )

        role = message.get("role")
        content = message.get("content")

        if role not in ("system", "user", "assistant"):
            raise ValueError(
                f"Esempio {index}, messaggio {msg_index}: "
                f"ruolo non supportato: {role}"
            )

        if not isinstance(content, str):
            raise ValueError(
                f"Esempio {index}, messaggio {msg_index}: "
                "content deve essere una stringa."
            )

        if not content.strip():
            raise ValueError(
                f"Esempio {index}, messaggio {msg_index}: content vuoto."
            )

    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"Esempio {index}: l'ultimo messaggio deve essere assistant."
        )

    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError(
            f"Esempio {index}: nessuna risposta assistant."
        )

    return messages


validated_records = []
for i, record in enumerate(records):
    validated_records.append(validate_messages(record, i))

print("Validazione conversazioni superata.")


# ============================================================
# 8. PREPROCESSING E CREAZIONE LABEL
# ============================================================

print("\nPreprocessing dataset...")

def preprocess_conversation(messages, index):
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )

    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]
    assistant_mask = result["assistant_masks"]

    if not (len(input_ids) == len(attention_mask) == len(assistant_mask)):
        raise ValueError(
            f"Esempio {index}: lunghezze dei token non corrispondenti."
        )

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            f"Esempio {index}: {len(input_ids)} token, "
            f"supera MAX_LENGTH={MAX_LENGTH}."
        )

    assistant_token_count = sum(assistant_mask)
    if assistant_token_count == 0:
        raise ValueError(
            f"Esempio {index}: maschera assistant vuota."
        )

    labels = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(input_ids, assistant_mask)
    ]

    if all(label == -100 for label in labels):
        raise ValueError(
            f"Esempio {index}: tutte le label sono ignorate."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


processed_records = []
total_tokens = 0
total_assistant_tokens = 0

for i, messages in enumerate(validated_records):
    item = preprocess_conversation(messages, i)
    processed_records.append(item)
    total_tokens += len(item["input_ids"])
    total_assistant_tokens += sum(label != -100 for label in item["labels"])

print("Preprocessing completato.")
print("Esempi:", len(processed_records))
print("Token totali:", total_tokens)
print("Token assistant:", total_assistant_tokens)

if total_assistant_tokens == 0:
    raise RuntimeError("Il dataset non contiene token assistant da addestrare.")


# ============================================================
# 9. CONTROLLO VISIVO DI UN ESEMPIO REALE
# ============================================================

sample = processed_records[0]
sample_target = [
    token_id
    for token_id, label in zip(sample["input_ids"], sample["labels"])
    if label != -100
]

print("\nEsempio reale del dataset:")
print(
    tokenizer.decode(
        sample_target,
        clean_up_tokenization_spaces=False
    )[:1500]
)


# ============================================================
# 10. TRAIN / VALIDATION SPLIT
# ============================================================

indices = list(range(len(processed_records)))
random.shuffle(indices)

validation_count = max(1, int(len(indices) * TEST_SIZE))
if validation_count >= len(indices):
    validation_count = len(indices) - 1

validation_indices = indices[:validation_count]
train_indices = indices[validation_count:]

train_records = [processed_records[i] for i in train_indices]
validation_records = [processed_records[i] for i in validation_indices]

print("\nTrain:", len(train_records))
print("Validation:", len(validation_records))


# ============================================================
# 11. DATA COLLATOR
# ============================================================

class AssistantOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for item in features:
            padding_length = max_len - len(item["input_ids"])
            input_ids = item["input_ids"] + [self.pad_token_id] * padding_length
            attention_mask = item["attention_mask"] + [0] * padding_length
            labels = item["labels"] + [-100] * padding_length

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


data_collator = AssistantOnlyCollator(tokenizer)


# ============================================================
# 12. CARICAMENTO MODELLO FP16
# ============================================================

print("\nCaricamento modello base FP16...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=True,
    torch_dtype=torch.float16,
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

print("Modello caricato.")
print("Dtype parametri:", next(model.parameters()).dtype)


# ============================================================
# 13. CONFIGURAZIONE LoRA
# ============================================================

print("\nConfigurazione LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()


# ============================================================
# 14. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
    seed=SEED,
)


# ============================================================
# 15. CREAZIONE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_records,
    eval_dataset=validation_records,
    data_collator=data_collator,
)


# ============================================================
# 16. TRAINING
# ============================================================

print("\n" + "=" * 60)
print("INIZIO TRAINING")
print("=" * 60)

trainer.train(
    resume_from_checkpoint=(None if not RESUME_FROM_CHECKPOINT else True)
)

print("\nTraining completato.")


# ============================================================
# 17. SALVATAGGIO ADAPTER LoRA
# ============================================================

print("\nSalvataggio adapter LoRA...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Adapter salvato in:", OUTPUT_DIR)


# ============================================================
# 18. VALUTAZIONE FINALE
# ============================================================

print("\nValutazione finale...")

eval_results = trainer.evaluate()
print("Risultati evaluation:", eval_results)

if "eval_loss" in eval_results:
    eval_loss = eval_results["eval_loss"]
    if eval_loss < 80:
        import math
        print("Perplexity:", math.exp(eval_loss))


# ============================================================
# 19. MERGE LoRA + MODELLO BASE FP16
# ============================================================

if DO_MERGE:
    print("\n" + "=" * 60)
    print("MERGE LoRA")
    print("=" * 60)

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Ricaricamento modello base FP16...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=True,
        torch_dtype=torch.float16,
    )

    print("Caricamento adapter...")

    merged_model = PeftModel.from_pretrained(
        base_model,
        OUTPUT_DIR,
    )

    print("Esecuzione merge...")

    merged_model = merged_model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)

    merged_model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    tokenizer.save_pretrained(MERGED_DIR)

    print("\nMerge completato.")
    print("Modello FP16 salvato in:", MERGED_DIR)

    del merged_model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 20. FINE
# ============================================================

print("\n" + "=" * 60)
print("PROCESSO COMPLETATO")
print("=" * 60)

print("Adapter LoRA:", OUTPUT_DIR)

if DO_MERGE:
    print("Modello merged FP16:", MERGED_DIR)

print("Training concluso correttamente.")