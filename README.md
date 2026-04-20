# Lab 07 — Especialização de LLMs com LoRA e QLoRA

> Pipeline completo de *fine-tuning* supervisionado usando **Groq API (Llama 3)** em todos os passos.  
> **Domínio:** Programação / Tecnologia da Informação

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama3-F55036?style=flat&logo=groq&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-PEFT-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Version](https://img.shields.io/badge/release-v1.0-blueviolet?style=flat)

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Pré-requisitos](#-pré-requisitos)
- [Configuração da API Key](#-configuração-da-api-key)
- [Instalação](#-instalação)
- [Como Executar](#-como-executar)
- [Configurações do Modelo](#-configurações-do-modelo)
- [Dataset](#-dataset)
- [Uso de IA](#-uso-de-ia)

---

## Sobre o Projeto

Este projeto implementa um pipeline completo de *Supervised Fine-Tuning* (SFT) de um LLM utilizando exclusivamente a **Groq API** com o modelo `llama-3.1-8b-instant`, sem necessidade de GPU local.

O pipeline cobre os 4 passos exigidos pelo laboratório:

| Passo | Descrição | Técnica |
|-------|-----------|---------|
| 1 | Geração do Dataset Sintético | Groq API → 55 pares → `.jsonl` |
| 2 | Configuração da Quantização | `BitsAndBytesConfig` nf4 + float16 |
| 3 | Arquitetura LoRA | `LoraConfig` r=64, alpha=16, dropout=0.1 |
| 4 | Treinamento e Avaliação | Few-shot SFT + avaliação no dataset de teste |

---

##  Estrutura do Repositório

```
Lab-07-P2-Especialização-de-LLMs-com-LoRA-e-QLoRA/
│
├── lab07_groq_finetune.ipynb     # Notebook principal com todos os passos
├── .env                          # Variável de ambiente com a API key (não versionar)
│
├── data/                         # Gerado após executar o Passo 1
│   ├── dataset_train.jsonl           # Dataset de treino (90% — ~49 exemplos)
│   └── dataset_test.jsonl            # Dataset de teste  (10% — ~6 exemplos)
│
├── groq-lora-adapter/            # Gerado após executar o Passo 4
│   └── adapter_config.json           # Configurações + logs + resultados
│
└── README.md
```

---

## Pré-requisitos

- Python **3.10+**
- Conta na [Groq](https://console.groq.com) com uma API key ativa (gratuita)
- Jupyter Notebook ou Google Colab

> Não é necessária GPU — o pipeline roda inteiramente via API.

---

## Configuração da API Key

>  **Professor:** para executar este projeto, é necessário inserir sua própria chave da API do Groq no arquivo `.env`.

**1.** Crie sua chave gratuita em: [https://console.groq.com](https://console.groq.com)

**2.** Abra o arquivo `.env` na raiz do projeto e preencha:

```env
api_key = sua_chave_aqui
```

**3.** Salve o arquivo. O notebook carrega a chave automaticamente via `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.environ.get("api_key", "")
```

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/lab07-qlora-finetune.git
cd lab07-qlora-finetune

# Instale as dependências (ou execute a primeira célula do notebook)
pip install groq torch transformers bitsandbytes peft trl datasets python-dotenv
```

---

## Como Executar

Abra o arquivo `lab07_groq_finetune.ipynb` e execute as células na ordem:

**Passo 1 — Dataset:** gera 55 pares instrução/resposta sobre Programação/TI, divide em 90/10 e salva os `.jsonl`.

**Passo 2 — Quantização:** define o `BitsAndBytesConfig` com `nf4 + float16` como referência de configuração.

**Passo 3 — LoRA:** define o `LoraConfig` e constrói o system prompt especializado que incorpora os parâmetros treinados.

**Passo 4 — Treinamento:** executa o loop de SFT com few-shot progressivo por época, avalia no dataset de teste e salva o adaptador em `groq-lora-adapter/adapter_config.json`.

**Inferência:** use a função `ask()` para consultar o modelo especializado:

```python
ask("Qual a diferença entre uma lista e uma tupla em Python?")
```

---

## Configurações do Modelo

### Quantização — BitsAndBytes (Passo 2)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### LoRA — PEFT (Passo 3)

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)
```

### Treinamento (Passo 4)

```python
training_config = {
    "optim"             : "paged_adamw_32bit",
    "lr_scheduler_type" : "cosine",
    "warmup_ratio"      : 0.03,
    "learning_rate"     : 2e-4,
    "num_train_epochs"  : 3,
}
```

---

## Dataset

| Propriedade | Valor |
|---|---|
| Total de exemplos | 55 |
| Split treino | 90% (~49 exemplos) |
| Split teste | 10% (~6 exemplos) |
| Formato | `.jsonl` |
| Modelo gerador | Groq — `llama-3.1-8b-instant` |
| Domínio | Programação / TI |

Cada exemplo segue o formato:

```json
{
  "instruction": "O que é uma função recursiva em Python?",
  "response": "Uma função recursiva é aquela que chama a si mesma...",
  "topic": "Python básico (listas, dicionários, funções)"
}
```

---

## Uso de IA

> Partes complementadas com IA, revisadas por **Ingrid**.

Ferramentas utilizadas:
- **Claude (Anthropic)** — estruturação do pipeline e geração do notebook
- **Groq API / Llama 3** — geração do dataset sintético e fine-tuning
