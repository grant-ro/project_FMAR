
import json
import sys
import subprocess
from pathlib import Path

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)



BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"

#Crea, se non esiste, la cartella models
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Cartella llama.cpp
LLAMA_CPP_DIR = PROJECT_DIR / "llama.cpp"

# Dataset
TRAINING_INPUT_DIR = (
    PROJECT_DIR / "data" / "input" / "training"
)

#Crea, se non esiste, la cartella TRAINING_INPUT_DIR
TRAINING_INPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPLETE_DATASET = (
    TRAINING_INPUT_DIR /
    "llama_training_dataset.json"
)


# Output training
TRAINING_OUTPUT_DIR = (
    PROJECT_DIR  / "data" / "output" / "training"
)

#Crea, se non esiste, la cartella TRAINING_OUTPUT_DIR
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_OUTPUT_DIR = (
    TRAINING_OUTPUT_DIR / "llama32-multitask"
)

#Cartella che conterrà i parametri LoRA addestrati
ADAPTER_OUTPUT_DIR = (
    TRAINING_OUTPUT_DIR / "llama32-multitask-adapter"
)

#Cartella che conterrà il modello completo, dopo l'unione dei parametri addestrati
MERGED_MODEL_DIR = (
    TRAINING_OUTPUT_DIR / "llama32-multitask-merged"
)

# GGUF finale
GGUF_OUTPUT_FILE = (
    MODELS_DIR / "llama32-multitask.gguf"
)



####  CONFIGURAZIONE MODELLO ####
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LENGTH = 2048

#Il 20% degli esempi è destinato alla validazione, l'80% all'addestramento
TEST_SIZE = 0.20 

RANDOM_SEED = 42

# Parametri LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Parametri training
EPOCHS = 3
LEARNING_RATE = 2e-4 #quanto vengono aggiornati i parametri addestrabili a ogni passo di ottimizzazione.

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

# Numero di processi per il caricamento dei dati
DATASET_NUM_PROC = 1



####  VERIFICA AMBIENTE  ####

print("\n======================================")
print("       AVVIO TRAINING LORA")
print("======================================")

print("PyTorch:", torch.__version__)
print("CUDA disponibile:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if not COMPLETE_DATASET.exists():
    raise FileNotFoundError(
        f"Dataset non trovato: {COMPLETE_DATASET}"
    )

if not LLAMA_CPP_DIR.exists():
    raise FileNotFoundError(
        f"Cartella llama.cpp non trovata: {LLAMA_CPP_DIR}\n"
        "Clona llama.cpp nella cartella principale del progetto."
    )

CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

if not CONVERT_SCRIPT.exists():
    raise FileNotFoundError(
        f"Script di conversione non trovato: {CONVERT_SCRIPT}"
    )


#### CARICAMENTO TOKENIZER ####
#il tokenizer permette di trasformare il testo in token da cui apprendere

print("\n[1/8] Caricamento tokenizer...")

#carica il tokenizer associato al nostro modello di riferimento
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"



#### CARICAMENTO DATASET ####
print("\n[2/8] Caricamento dataset...")

dataset = load_dataset(
    "json",
    data_files=str(COMPLETE_DATASET),
    split="train",
)

print("Numero totale esempi:", len(dataset))

if len(dataset) < 2:
    raise ValueError(
        "Il dataset deve contenere almeno due esempi."
    )

# Suddivisione training / validation
dataset_split = dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=RANDOM_SEED,
    shuffle=True, #gli esempi vengono mescolati prima della suddivisione.
)

train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print("Esempi training:", len(train_dataset))
print("Esempi validation:", len(eval_dataset))


#### PREPARAZIONE DATI  ####
print("\n[3/8] Tokenizzazione dataset...")

def preprocess_example(example):

    messages = example["messages"]

    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(
            "Ogni esempio deve contenere una lista messages valida."
        )

    # L'ultimo messaggio deve essere quello dell'assistente
    if messages[-1]["role"] != "assistant":
        raise ValueError(
            "L'ultimo messaggio deve avere role='assistant'."
        )

    # Conversazione completa
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Conversazione prima della risposta dell'assistente
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenizzazione
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    # Controllo per evitare esempi senza token di risposta
    if len(prompt_ids) >= len(full_ids):
        raise ValueError(
            "Esempio senza token di risposta utilizzabili."
        )

    # Loss soltanto sulla risposta dell'assistente
    labels = [-100] * len(prompt_ids)
    labels += full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


train_dataset = train_dataset.map(
    preprocess_example,
    remove_columns=train_dataset.column_names,
    num_proc=DATASET_NUM_PROC,
)

eval_dataset = eval_dataset.map(
    preprocess_example,
    remove_columns=eval_dataset.column_names,
    num_proc=DATASET_NUM_PROC,
)

print("Tokenizzazione completata.")



#### CARICAMENTO MODELLO ####
print("\n[4/8] Caricamento modello base...")

if torch.cuda.is_available():

    if torch.cuda.is_bf16_supported():
        MODEL_DTYPE = torch.bfloat16
    else:
        MODEL_DTYPE = torch.float16

else:
    MODEL_DTYPE = torch.float32


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=MODEL_DTYPE,
)

model.config.use_cache = False

model.gradient_checkpointing_enable()

model.enable_input_require_grads()


#### CONFIGURAZIONE LORA ####
print("\n[5/8] Configurazione LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()



#### CONFIGURAZIONE TRAINING ####
print("\n[6/8] Preparazione training...")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)


training_args = TrainingArguments(

    output_dir=str(LORA_OUTPUT_DIR),

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,
    warmup_steps=0.03,

    logging_steps=10,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    save_total_limit=2,

    fp16=(
        torch.cuda.is_available()
        and MODEL_DTYPE == torch.float16
    ),

    bf16=(
        torch.cuda.is_available()
        and MODEL_DTYPE == torch.bfloat16
    ),

    gradient_checkpointing=True,

    optim="adamw_torch",

    report_to="none",

    remove_unused_columns=False,

    seed=RANDOM_SEED,
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    data_collator=data_collator,
)



#### AVVIO TRAINING ####

print("\n======================================")
print("       INIZIO ADDESTRAMENTO")
print("======================================")

#Esecuzione dell'addestranìmento
trainer.train()

print("\nTraining completato!")

# Salvataggio adapter LoRA
trainer.model.save_pretrained(
    str(ADAPTER_OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(ADAPTER_OUTPUT_DIR)
)

print("Adapter LoRA salvato in:")
print(ADAPTER_OUTPUT_DIR)

# Valutazione finale
evaluation_results = trainer.evaluate()

print("\nRisultati validation:")
print(evaluation_results)

# Salva metriche
with open(
    TRAINING_OUTPUT_DIR / "evaluation_results.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        evaluation_results,
        f,
        indent=4,
        ensure_ascii=False,
    )


#### LIBERAZIONE MEMORIA ####
print("\n[7/8] Preparazione merge modello...")

del trainer
del model

import gc

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()



####  MERGE MODELLO BASE + ADAPTER LORA #### 
print("\nUnione modello base e adapter LoRA...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=MODEL_DTYPE,
)

lora_model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_OUTPUT_DIR),
)

merged_model = lora_model.merge_and_unload()

merged_model.config.use_cache = True

# Salvataggio modello completo Hugging Face
merged_model.save_pretrained(
    str(MERGED_MODEL_DIR),
    safe_serialization=True,
)

tokenizer.save_pretrained(
    str(MERGED_MODEL_DIR)
)

print("Modello unito salvato in:")
print(MERGED_MODEL_DIR)

del merged_model
del lora_model
del base_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


#### CONVERSIONE IN FORMATO GGUF #### 
print("\n[8/8] Conversione modello in GGUF...")

subprocess.run(
    [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(MERGED_MODEL_DIR),
        "--outfile",
        str(GGUF_OUTPUT_FILE),
        "--outtype",
        "f16",
    ],
    check=True,
)

# Controllo file finale
if not GGUF_OUTPUT_FILE.exists():
    raise RuntimeError(
        "Conversione GGUF terminata senza creare il file."
    )

print("\n======================================")
print("       PROCESSO COMPLETATO")
print("======================================")

print("\nModello GGUF creato correttamente!")

print("Percorso:")
print(GGUF_OUTPUT_FILE)

print("\nDimensione file:")

size_gb = GGUF_OUTPUT_FILE.stat().st_size / (1024 ** 3)

print(f"{size_gb:.2f} GB")

print("\nPuoi ora utilizzare il modello con llama.cpp.")
