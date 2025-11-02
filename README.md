# 🧬 Pima Indians Diabetes Prediction: Otimização de Recall para Triagem Clínica

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange?logo=scikit-learn)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-blueviolet?logo=xgboost)](https://xgboost.ai/)
[![Metodologia](https://img.shields.io/badge/Metodologia-CRISP--DM-lightgrey)](https://www.ibm.com/topics/crisp-dm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Visão Geral do Projeto

Este projeto de Machine Learning (ML) tem como objetivo principal desenvolver um modelo de classificação binária altamente sensível para prever o risco de diabetes em pacientes do Pima Indian, utilizando o dataset Pima Indians Diabetes.

Dada a natureza clínica do problema, a prioridade máxima foi a **minimização de Falsos Negativos (FN)** – reduzir o erro de deixar um paciente diabético sem diagnóstico. A estratégia resultou em um modelo que alcançou **94.4% de Recall**, o que é crucial para uma ferramenta de triagem segura.

### 🌟 Modelo Final e Performance Chave

| Métrica                   | Valor Final (Conjunto de Teste)     | Justificativa Clínica                                                                                                     |
| :------------------------ | :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Recall (Classe 1)**     | **0.9444**                          | **Sucesso Crítico:** Indica que 94.4% dos pacientes diabéticos foram corretamente detectados, minimizando o risco à vida. |
| **Falsos Negativos (FN)** | **3 em 54 casos**                   | Redução de 86% dos FN críticos em relação ao Baseline.                                                                    |
| **AUC-ROC**               | **0.8116**                          | Forte capacidade de discriminação entre as classes.                                                                       |
| **Modelo**                | **XGBoost Classifier** (Fine-Tuned) | Modelo mais robusto para a generalização de alta sensibilidade.                                                           |

---

---

## 🚀 Configuração e Execução do Projeto

Este projeto usa **Python 3.12** e o **gerenciador de dependências `uv`**.

### 1) Clonar o repositório

```bash
git clone https://github.com/luisconcha/model_diabetes.git
cd model_diabetes
```

### 2) Instalar o Python (se necessário)

```bash
uv python install 3.12
```

### 3) Criar o ambiente e instalar dependências

```bash
uv sync
```

### 4) Verificar o ambiente

```bash
uv run python --version
uv run python -c "import sklearn, xgboost, pandas; print('Tudo OK')"
```

### 5) Executar o notebook

```bash
Se está utilizando o VSCode, é necessário ter o plugin **Jupyter** instalado.
Se preferir, o notebook também pode ser aberto no **JupyterLab**.
```

### 6) Estrutura de dados

-   Dataset: `datasets/medical/diabetes.csv`
-   Artefatos: `deployment_artifacts/`
    -   `diabetes_imputer_median.joblib`
    -   `diabetes_scaler.joblib`
    -   `diabetes_xgb_model.joblib`

### 8) Alternativa com pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r <(uv export --no-dev --format requirements-txt)
```

---

## 💡 Abordagem e Estratégia Metodológica

O projeto seguiu rigorosamente as etapas do CRISP-DM, com foco especial na preparação de dados e otimização de métricas:

### 1. Preparação de Dados e Feature Engineering

-   **Tratamento de NaNs/Zeros:** Zeros em features como `glucose` e `insulin` foram tratados como NaNs e imputados pela **Mediana** para garantir robustez contra outliers.
-   **Seleção de Features:** O modelo final foi simplificado para **8 features clínicas** (as flags de imputação foram removidas após análise de baixa importância).

### 2. Análise Exploratória de Dados (EDA)

-   Confirmação de desbalanceamento de classes (65% e 35%).
-   Identificação dos preditores mais fortes: **`glucose`** (o mais forte), **`bmi`**, e **`age`**.
-   Validação da necessidade de _Feature Scaling_ (Padronização) devido às diferentes escalas (e.g., `insulin` vs. `age`).

### 3. Modelagem e Otimização Crítica

-   **Prevenção de Data Leakage:** A padronização (`StandardScaler`) foi aplicada **apenas** no conjunto de Treinamento (`X_train`) e depois replicada nos conjuntos de Validação e Teste.
-   **Otimização do Recall:** Utilização de `GridSearchCV` no XGBoost, com o parâmetro `scoring='recall'` (focado na Classe 1) e ajuste do `scale_pos_weight`.
    -   Esta otimização foi a chave para o sucesso, transformando o Recall de **0.59 (fragilidade)** para **0.9444 (segurança)**.

---

## 💾 Estrutura do Repositório e Deployment

O projeto é modular e está estruturado para ser facilmente transferido para um ambiente de produção (API REST).

### Estrutura

```bash
model_diabetes/
├── datasets/
│   └── medical/
│       └── diabetes.csv
├── deployment_artifacts/
│   ├── diabetes_imputer_median.joblib
│   ├── diabetes_scaler.joblib
│   └── diabetes_xgb_model.joblib
├── eda/
│   └── notebook/
│       └── model_diabetes.ipynb
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

---

### Pipeline de Inferência

O modelo em produção utiliza um pipeline de inferência serializado:

**Dados Brutos (8 Features) → Imputação (medianas) → Padronização (Scaler salvo) → XGBoost → Diagnóstico e Probabilidade.**

O objeto `diabetes_scaler.joblib` garante que os dados de entrada sejam processados **exatamente** como no treinamento.

---

## ⚠️ Disclaimer

> O diagnóstico produzido pelo modelo é um auxílio computacional para triagem e **não substitui avaliação médica presencial**.

---

## 🤝 Créditos e Contato

Este projeto foi desenvolvido por **Luis Alberto Concha Curay** como um estudo de caso aprofundado em Machine Learning e engenharia de software para avaliação de risco clínico.

**Desenvolvedor Principal:**

-   **Nome:** Luis Alberto Concha Curay
-   **LinkedIn:** [https://www.linkedin.com/in/luis-alberto-concha-curay/](https://www.linkedin.com/in/luis-alberto-concha-curay/)
-   **GitHub:** [https://github.com/luisconcha](https://github.com/luisconcha)

**Tecnologias Utilizadas:**

-   `Python`
-   `pandas`, `numpy`
-   `scikit-learn` (Regressão Logística, StandardScaler, Metrics)
-   `XGBoost` (Modelo Final)
-   `Plotly` (Visualizações Interativas)
-   `joblib` (Serialização para Deployment)

---
