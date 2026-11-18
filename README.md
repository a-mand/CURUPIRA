# CURUPIRA: simulação em Python

Código inicial para as simulações do projeto: estações IoT que preveem o clima,
detectam o drift de um evento extremo e aprendem juntas via aprendizado federado (FL)
com atualizações quantizadas.

> Os dados padrão são **sintéticos** e servem para testar o código. Os resultados
> do artigo devem vir dos dados reais do INMET (seção 4).

## 1. Instalação

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

## 2. Rodar

```bash
python run_experiment.py --seeds 10
python run_experiment.py --seeds 10 --bits 4
```

Saídas na pasta `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `resultados.csv` | todas as métricas, por semente e estratégia |
| `resumo.csv` | média ± IC 95%, redução do RMSE e p-valor do Wilcoxon |
| `curva_rmse.png` | erro diário nos nós do epicentro |

## 3. O que cada arquivo faz

| Arquivo | Papel | Liga com o projeto |
|---|---|---|
| `curupira/data.py` | gera ou lê os dados e monta as janelas (12 h de histórico → temperatura daqui a 3 h) | A1 |
| `curupira/model.py` | MLP pequena (~1.300 parâmetros, 5 kB) em NumPy | A2, M1 |
| `curupira/drift.py` | detector Page-Hinkley (leve, cabe no ESP32) | A4, M2 |
| `curupira/fl.py` | agregação, peso por severidade e quantização int8/int4 | A5, M3, M4 |
| `run_experiment.py` | roda todas as estratégias e calcula as métricas | M1–M4 |

### Como a simulação funciona, dia a dia

1. **Pré-treino:** FedAvg sobre os 300 dias "normais".
2. **Período de teste:** a cada hora, cada nó **prevê e depois observa** o valor real
   (avaliação prequencial). O erro alimenta o detector de drift.
3. **No fim de cada dia**, cada estratégia atualiza (ou não) os modelos:

| Estratégia | O que faz |
|---|---|
| `B1_estatico` | nunca atualiza |
| `B2_local` | cada nó retreina sozinho, sem colaboração |
| `B3_fedavg` | todos os nós mandam atualização em float32, com pesos iguais |
| `P_uniforme` | proposta sem peso por severidade (testa a hipótese H1) |
| `P_float32` | proposta sem quantização (testa a hipótese H2) |
| `Proposta` | só os nós em alerta de drift participam, com peso pela severidade e envio em int8; os demais só adotam o modelo global se ele errar menos |

**Severidade** = erro médio das últimas 24 h ÷ erro normal do nó. Peso = `min(4, 1 + 2·(sev − 1))`.
