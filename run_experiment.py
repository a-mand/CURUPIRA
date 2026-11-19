"""Experimento principal do CURUPIRA (simulação) com aprendizado federado hierárquico por microrregiões.

Arquitetura em dois níveis:
  estação (ESP32)  --LoRa-->  agregador da microrregião (gateway)  --backhaul-->  servidor estadual

Estratégias comparadas (várias sementes):
  B1_estatico      modelo treinado uma vez, nunca atualizado
  B2_local         cada estação ajusta o próprio modelo todo dia, sem colaboração
  B3_fedavg        FedAvg plano: todas as estações -> servidor estadual, todo dia, float32
  B4_hier          FedAvg hierárquico clássico: todas as estações -> microrregião (diário),
                   microrregiões -> estado (a cada K dias), pesos iguais, float32
  P_plano          proposta SEM hierarquia (drift + severidade + int8, um único servidor)
  P_uniforme       proposta hierárquica SEM peso por severidade        (ablação de H1)
  P_float32        proposta hierárquica SEM quantização                (ablação de H2)
  Proposta         drift on-device + FL hierárquico ponderado por severidade + int8

Uso:  python run_experiment.py --seeds 10
      python run_experiment.py --seeds 10 --bits 4 --K 2
Saídas em resultados/: resultados.csv, resumo.csv, por_microrregiao.csv, curva_rmse.png
"""
import argparse, os, time
import numpy as np
import pandas as pd

from curupira.data import gerar_microrregioes, Normalizador, janelas
from curupira.model import MLP
from curupira.drift import PageHinkley
from curupira.fl import quantizar, agregar, peso_severidade, treino_local

# nome: (hierárquico, disparo por drift, peso por severidade, bits)
CONFIG = {
    "B3_fedavg":  (False, False, False, 32),
    "B4_hier":    (True,  False, False, 32),
    "P_plano":    (False, True,  True,  None),
    "P_uniforme": (True,  True,  False, None),
    "P_float32":  (True,  True,  True,  32),
    "Proposta":   (True,  True,  True,  None),
}
ESTRATEGIAS = ["B1_estatico", "B2_local"] + list(CONFIG)


def preparar(seed, args):
    estacoes, eventos, mr_de, papel = gerar_microrregioes(
        estacoes_por_mr=args.estacoes_por_mr, dias=args.dias, inicio_teste=args.inicio_teste, seed=seed)
    h_teste = args.inicio_teste * 24
    norm = Normalizador().fit([df.iloc[:h_teste] for df in estacoes.values()])
    dados = {s: janelas(df, norm, W=args.W, H=args.H) for s, df in estacoes.items()}
    return dados, eventos, mr_de, papel, norm, h_teste


def pre_treino(dados, h_teste, args, seed):
    """FedAvg sobre o período normal (antes do teste) para obter o modelo inicial."""
    n_in = next(iter(dados.values()))[0].shape[1]
    modelo = MLP(n_in, n_hid=args.hid, seed=seed)
    theta = modelo.get()
    for r in range(args.rodadas_pre):
        deltas = []
        for s, (X, y, t) in dados.items():
            m = t < h_teste
            deltas.append(treino_local(modelo, theta, X[m], y[m], epochs=1, lr=2e-3, seed=seed + r))
        theta = agregar(theta, deltas, [1] * len(deltas))
    ref = {}                                          # erro normal de cada estação
    modelo.set(theta)
    for s, (X, y, t) in dados.items():
        m = (t >= h_teste - 14 * 24) & (t < h_teste)
        ref[s] = float(np.mean(np.abs(modelo.predict(X[m]) - y[m])))
    return modelo, theta, ref


def _erro(modelo, theta, X, y):
    modelo.set(theta)
    return float(np.mean(np.abs(modelo.predict(X) - y)))


def simular(estrategia, dados, eventos, mr_de, h_teste, modelo, theta0, ref, args, seed):
    nos = list(dados.keys())
    local = {s: theta0.copy() for s in nos}           # modelo em uso em cada estação
    det = {s: PageHinkley(delta=args.ph_delta, lam=args.ph_lambda) for s in nos}
    alerta_ate = {s: -1 for s in nos}
    alarmes = {s: [] for s in nos}
    registros = []                                     # (estação, hora, erro)
    lora_up = lora_down = backhaul = 0

    if estrategia in CONFIG:
        hier, por_drift, por_sev, bits = CONFIG[estrategia]
        bits = bits or args.bits
        # plano = uma "região" só, ligada direto ao servidor
        regioes = {}
        for s in nos:
            regioes.setdefault(mr_de[s] if hier else "estado", []).append(s)
        theta_r = {r: theta0.copy() for r in regioes}  # modelo de cada microrregião
        theta_e = theta0.copy()                         # modelo estadual
        sev_acum = {r: [] for r in regioes}             # severidades no ciclo de K dias
        ativas_ontem = set()

    n_dias = (len(next(iter(dados.values()))[2]) + args.W + args.H) // 24 - h_teste // 24
    for d in range(n_dias):
        h_ini, h_fim = h_teste + d * 24, h_teste + (d + 1) * 24
        buffer = {}
        # 1) previsão prequencial do dia + detector de drift em cada estação
        for s in nos:
            X, y, t = dados[s]
            m = (t >= h_ini) & (t < h_fim)
            if not m.any():
                continue
            modelo.set(local[s])
            err = modelo.predict(X[m]) - y[m]
            for hh, e in zip(t[m], err):
                registros.append((s, hh, e))
                if det[s].update(abs(e)):
                    alarmes[s].append(hh)
                    alerta_ate[s] = d + args.dias_alerta
            mb = (t >= h_fim - args.buffer_h) & (t < h_fim)
            buffer[s] = (X[mb], y[mb], float(np.mean(np.abs(err[-24:]))) / ref[s])
        if not buffer:
            break

        # 2) atualizações
        if estrategia == "B1_estatico":
            continue
        if estrategia == "B2_local":
            for s, (Xb, yb, _) in buffer.items():
                local[s] = local[s] + treino_local(modelo, local[s], Xb, yb, epochs=args.ep, lr=args.lr, seed=seed + d)
            continue

        # nível 1: estações -> agregador da microrregião (via LoRa)
        ativas_hoje = set()
        for r, membros in regioes.items():
            membros = [s for s in membros if s in buffer]
            part = [s for s in membros if (d <= alerta_ate[s]) or not por_drift]
            if not part:
                continue
            ativas_hoje.add(r)
            deltas, pesos = [], []
            for s in part:
                Xb, yb, sev = buffer[s]
                delta = treino_local(modelo, theta_r[r], Xb, yb, epochs=args.ep, lr=args.lr, seed=seed + d)
                dq, nb = quantizar(delta, bits)
                deltas.append(dq); lora_up += nb
                pesos.append(peso_severidade(sev, args.alpha, args.w_max) if por_sev else 1.0)
                sev_acum[r].append(sev)
            novo = agregar(theta_r[r], deltas, pesos)
            _, nb_down = quantizar(novo - theta_r[r], bits)
            theta_r[r] = novo
            for s in membros:                          # difusão para as estações da microrregião
                lora_down += nb_down
                if not por_drift or s in part:
                    local[s] = theta_r[r].copy()
                else:                                  # porta de adoção: só aceita se melhorar
                    Xb, yb, _ = buffer[s]
                    if _erro(modelo, theta_r[r], Xb, yb) < _erro(modelo, local[s], Xb, yb):
                        local[s] = theta_r[r].copy()

        # nível 2: microrregiões -> servidor estadual (backhaul), a cada K dias
        # (por evento: uma microrregião que acabou de entrar em alerta antecipa a rodada estadual)
        nova_mr_em_alerta = por_drift and args.rodada_por_evento and bool(ativas_hoje - ativas_ontem)
        ativas_ontem = ativas_hoje
        if hier and ((d + 1) % args.K == 0 or nova_mr_em_alerta):
            ativas = [r for r in regioes if sev_acum[r]]
            if ativas:
                deltas = []
                for r in ativas:
                    dq, nb = quantizar(theta_r[r] - theta_e, bits)
                    deltas.append(dq); backhaul += nb
                pesos = [peso_severidade(float(np.mean(sev_acum[r])), args.alpha, args.w_max) if por_sev else 1.0
                         for r in ativas]
                theta_e = agregar(theta_e, deltas, pesos)
                _, nb = quantizar(theta_e, bits)
                backhaul += nb * len(regioes)
                for r in regioes:
                    # microrregiões sem alerta recebem o modelo estadual "pré-aquecido": se o
                    # evento chegar até elas, as estações já partem do que o epicentro aprendeu
                    if not por_drift or r not in ativas or not por_sev:
                        theta_r[r] = theta_e.copy()
                    else:                              # microrregião em evento: mistura regional + estadual
                        theta_r[r] = ((1 - args.mistura) * theta_r[r] + args.mistura * theta_e).astype(np.float32)
                    if not por_drift:
                        for s in regioes[r]:
                            local[s] = theta_r[r].copy()
            sev_acum = {r: [] for r in regioes}

    return registros, alarmes, lora_up, lora_down, backhaul


def metricas(registros, alarmes, eventos, mr_de, papel, h_teste, sd_temp, lora_up, lora_down, backhaul):
    df = pd.DataFrame(registros, columns=["no", "hora", "erro"])
    df["erro_c"] = df["erro"] * sd_temp
    df["mr"] = df["no"].map(mr_de)
    df["papel"] = df["mr"].map(papel)
    em_ev = np.zeros(len(df), bool)
    inicio_ev = np.zeros(len(df), bool)               # primeiras 72 h de evento em cada estação
    for s, ev in eventos.items():
        if ev:
            ms = df["no"].values == s
            em_ev |= ms & (df["hora"].values >= ev[0]) & (df["hora"].values < ev[1])
            inicio_ev |= ms & (df["hora"].values >= ev[0]) & (df["hora"].values < ev[0] + 72 + 7 * 24)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
    atrasos, falsos = [], 0
    for s, ev in eventos.items():
        a = np.array(alarmes[s])
        if ev:
            depois = a[a >= ev[0]]
            atrasos.append((depois[0] - ev[0]) if len(depois) else np.nan)
            falsos += int(np.sum(a < ev[0]))
        else:
            falsos += len(a)
    out = {
        "rmse_evento_C": rmse(df.loc[em_ev, "erro_c"].values),
        "rmse_epicentro_C": rmse(df.loc[em_ev & (df.papel == "epicentro"), "erro_c"].values),
        "rmse_vizinhas_C": rmse(df.loc[em_ev & (df.papel == "vizinha"), "erro_c"].values),
        "rmse_vizinhas_inicio_C": rmse(df.loc[inicio_ev & (df.papel == "vizinha"), "erro_c"].values),
        "rmse_fora_C": rmse(df.loc[df.papel == "fora", "erro_c"].values),
        "atraso_deteccao_h": float(np.nanmedian(atrasos)) if not np.all(np.isnan(atrasos)) else np.nan,
        "falsos_alarmes": falsos,
        "kB_lora_up": lora_up / 1024, "kB_lora_down": lora_down / 1024, "kB_backhaul": backhaul / 1024,
    }
    por_mr = df[em_ev | (df.papel == "fora")].groupby("mr")["erro_c"].apply(lambda e: rmse(e.values)).to_dict()
    return out, por_mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--estacoes_por_mr", type=int, default=4)
    ap.add_argument("--dias", type=int, default=420)
    ap.add_argument("--inicio_teste", type=int, default=300)
    ap.add_argument("--W", type=int, default=12)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--rodadas_pre", type=int, default=15)
    ap.add_argument("--ep", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--buffer_h", type=int, default=72)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--K", type=int, default=3, help="dias entre rodadas do nível estadual")
    ap.add_argument("--rodada_por_evento", type=int, default=1, help="1 = antecipa a rodada estadual quando uma microrregião entra em alerta")
    ap.add_argument("--mistura", type=float, default=0.5, help="peso do modelo estadual nas microrregiões em evento")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--w_max", type=float, default=4.0)
    ap.add_argument("--dias_alerta", type=int, default=10)
    ap.add_argument("--ph_delta", type=float, default=0.05)
    ap.add_argument("--ph_lambda", type=float, default=8.0)
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    linhas, linhas_mr, curvas = [], [], {}
    for seed in range(args.seeds):
        t0 = time.time()
        dados, eventos, mr_de, papel, norm, h_teste = preparar(seed, args)
        modelo, theta0, ref = pre_treino(dados, h_teste, args, seed)
        if seed == 0:
            print(f"{len(dados)} estações em {len(papel)} microrregiões; modelo com {modelo.n_params} "
                  f"parâmetros ({modelo.n_params * 4 / 1024:.1f} kB em float32)")
        for est in ESTRATEGIAS:
            reg, alarmes, lu, ld, bh = simular(est, dados, eventos, mr_de, h_teste, modelo, theta0, ref, args, seed)
            m, por_mr = metricas(reg, alarmes, eventos, mr_de, papel, h_teste, norm.sd[0], lu, ld, bh)
            linhas.append({"seed": seed, "estrategia": est, **m})
            linhas_mr += [{"seed": seed, "estrategia": est, "microrregiao": k, "papel": papel[k], "rmse_C": v}
                          for k, v in por_mr.items()]
            if seed == 0:
                curvas[est] = reg
        print(f"[seed {seed}] concluída em {time.time() - t0:.0f} s")

    res = pd.DataFrame(linhas)
    res.to_csv(f"{args.saida}/resultados.csv", index=False)
    pmr = pd.DataFrame(linhas_mr)
    pmr.groupby(["microrregiao", "papel", "estrategia"])["rmse_C"].mean().unstack().round(3) \
        .to_csv(f"{args.saida}/por_microrregiao.csv")

    from scipy import stats
    cols = ["rmse_evento_C", "rmse_epicentro_C", "rmse_vizinhas_C", "rmse_vizinhas_inicio_C",
            "rmse_fora_C", "kB_lora_up", "kB_backhaul", "atraso_deteccao_h", "falsos_alarmes"]
    resumo = []
    for est in ESTRATEGIAS:
        r = res[res.estrategia == est]
        linha = {"estrategia": est}
        for c in cols:
            v = r[c].dropna().values
            ic = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            linha[c] = f"{v.mean():.3f} ± {ic:.3f}" if len(v) else "n/a"
        if est != "Proposta" and args.seeds >= 5:
            a = res[res.estrategia == "Proposta"]["rmse_evento_C"].values
            b = r["rmse_evento_C"].values
            linha["p_wilcoxon_evento"] = f"{stats.wilcoxon(a, b).pvalue:.4f}"
            linha["reducao_rmse_evento_%"] = f"{100 * (b.mean() - a.mean()) / b.mean():.1f}"
            a2 = res[res.estrategia == "Proposta"]["rmse_vizinhas_inicio_C"].values
            b2 = r["rmse_vizinhas_inicio_C"].values
            linha["p_wilcoxon_vizinhas_inicio"] = f"{stats.wilcoxon(a2, b2).pvalue:.4f}"
        resumo.append(linha)
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(f"{args.saida}/resumo.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== RESUMO (média ± IC 95%) ===")
    print(resumo.to_string(index=False))
    if args.seeds < 5:
        print("(use --seeds 5 ou mais para o teste de Wilcoxon)")

    # gráfico: RMSE diário nas microrregiões do epicentro e nas vizinhas (seed 0)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, eventos, mr_de, papel, norm, h_teste = preparar(0, args)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, alvo in zip(axs, ["epicentro", "vizinha"]):
        nos_alvo = [s for s in mr_de if papel[mr_de[s]] == alvo]
        for est in ["B1_estatico", "B3_fedavg", "B4_hier", "P_plano", "Proposta"]:
            df = pd.DataFrame(curvas[est], columns=["no", "hora", "erro"])
            df = df[df.no.isin(nos_alvo)]
            df["dia"] = (df.hora - h_teste) // 24
            g = df.groupby("dia")["erro"].apply(lambda e: np.sqrt(np.mean((e * norm.sd[0]) ** 2)))
            ax.plot(g.index, g.values, label=est, lw=2.2 if est == "Proposta" else 1.1)
        ev = [eventos[s] for s in nos_alvo]
        ax.axvspan((min(e[0] for e in ev) - h_teste) / 24, (max(e[1] for e in ev) - h_teste) / 24,
                   color="orange", alpha=0.15, label="evento extremo")
        mrs = sorted({mr_de[s] for s in nos_alvo})
        ax.set_title(f"Microrregiões {alvo}s ({', '.join(mrs)})" if alvo == "vizinha"
                     else f"Microrregiões do epicentro ({', '.join(mrs)})", fontsize=10)
        ax.set_xlabel("dia do período de teste"); ax.grid(alpha=.3)
    axs[0].set_ylabel("RMSE diário (°C)"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{args.saida}/curva_rmse.png", dpi=150)
    print(f"\nArquivos salvos em {args.saida}/")


if __name__ == "__main__":
    main()
