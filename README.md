# CURUPIRA: simulação em Python

Código das simulações do projeto: estações IoT que preveem o clima, detectam o drift
de um evento extremo e aprendem juntas por **aprendizado federado hierárquico por
microrregiões**, com atualizações quantizadas:

```
estação (ESP32) --LoRa--> agregador da microrregião (gateway) --backhaul--> servidor estadual
```
## 1. Instalação

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Rodar

```bash
python run_experiment.py --seeds 10              # ~1,5 min
python run_experiment.py --seeds 10 --bits 4     # quantização de 4 bits
python run_experiment.py --seeds 10 --K 7        # rodada estadual semanal
python run_experiment.py --estacoes_por_mr 8     # mais estações por microrregião
```

Saídas na pasta `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `resultados.csv` | todas as métricas, por semente e estratégia |
| `resumo.csv` | média ± IC 95%, redução do RMSE e p-valor do Wilcoxon |
| `por_microrregiao.csv` | RMSE médio por microrregião e estratégia |
| `curva_rmse.png` | erro diário nas microrregiões do epicentro e nas vizinhas (figura para o artigo) |

## 3. O que cada arquivo faz

| Arquivo | Papel | Liga com o projeto |
|---|---|---|
| `curupira/data.py` | gera dados de 6 microrregiões × 4 estações (ou lê o INMET) e monta as janelas (12 h → temperatura daqui a 3 h) | A1 |
| `curupira/model.py` | MLP pequena (~1.300 parâmetros, 5 kB) em NumPy | A2, M1 |
| `curupira/drift.py` | detector Page-Hinkley (leve, cabe no ESP32) | A4, M2 |
| `curupira/fl.py` | agregação, peso por severidade e quantização int8/int4 | A5, M3, M4 |
| `run_experiment.py` → `simular` | FL em dois níveis: microrregião (diário) e estado (a cada K dias ou quando uma microrregião entra em alerta) | A5 |
| `run_experiment.py` | roda todas as estratégias e calcula as métricas | M1–M4 |

### Microrregiões do cenário sintético

| Microrregião (IBGE) | Papel no evento sintético |
|---|---|
| Santarém, Óbidos | epicentro (seca começa no dia 40, intensidade 100%) |
| Itaituba, Altamira | vizinhas (começa ~10 dias depois, intensidade ~60%) |
| Marabá, Belém | fora do evento |

> Os papéis são **hipotéticos**, só para testar o código. Com dados reais, eles saem da rotulagem dos eventos.

### Como a simulação funciona, dia a dia

1. **Pré-treino:** FedAvg sobre os 300 dias "normais".
2. **Teste:** a cada hora, cada estação **prevê e depois observa** o valor real (avaliação prequencial). O erro alimenta o detector de drift.
3. **No fim de cada dia:**
   - **Nível 1 (microrregião):** as estações mandam a atualização ao agregador da sua microrregião (LoRa).
   - **Nível 2 (estado):** a cada K dias, ou assim que uma microrregião nova entra em alerta, os agregadores mandam o modelo regional ao servidor estadual (backhaul). O modelo estadual volta para as microrregiões sem alerta, que ficam **"pré-aquecidas"** caso o evento chegue até elas.

| Estratégia | O que faz |
|---|---|
| `B1_estatico` | nunca atualiza |
| `B2_local` | cada estação retreina sozinha |
| `B3_fedavg` | FedAvg plano: todas as estações → estado, float32 |
| `B4_hier` | FedAvg hierárquico clássico (pesos iguais, todas participam, float32) |
| `P_plano` | proposta **sem** hierarquia |
| `P_uniforme` | proposta hierárquica sem peso por severidade (H1) |
| `P_float32` | proposta hierárquica sem quantização (H2) |
| `Proposta` | só estações em alerta participam; peso por severidade nos dois níveis; int8 |

**Severidade** = erro médio das últimas 24 h ÷ erro normal da estação. Peso = `min(4, 1 + 2·(sev − 1))`.

### Métricas principais

| Coluna | Significado |
|---|---|
| `rmse_epicentro_C` / `rmse_vizinhas_C` | erro durante o evento em cada grupo de microrregiões |
| `rmse_vizinhas_inicio_C` | erro nas vizinhas nos primeiros ~10 dias de evento (mede o "aviso antecipado") |
| `rmse_fora_C` | erro nas microrregiões fora do evento (a proposta não pode piorá-las) |
| `kB_lora_up` / `kB_backhaul` | tráfego no rádio LoRa e entre gateways e o servidor |

