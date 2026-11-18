"""Experimento principal do CURUPIRA (simulação).

Compara, com várias sementes:
  B1_estatico          modelo treinado uma vez, nunca atualizado
  B2_local             cada nó ajusta o próprio modelo todo dia, sem colaboração
  B3_fedavg            FedAvg diário com todos os nós, float32
  P_uniforme           proposta sem ponderação por severidade (ablação de H1)
  P_float32            proposta sem quantização (ablação de H2)
  Proposta             drift on-device + FL ponderado por severidade + int8

Uso:  python run_experiment.py --seeds 5
      python run_experiment.py --seeds 10 --bits 4
Saídas em resultados/: resultados.csv, resumo.csv, curva_rmse.png
"""
import argparse, os, time
import numpy as np
import pandas as pd

from curupira.data import gerar_estacoes, Normalizador, janelas
from curupira.model import MLP
from curupira.drift import PageHinkley
from curupira.fl import quantizar, agregar, peso_severidade, treino_local

ESTRATEGIAS = ["B1_estatico", "B2_local", "B3_fedavg", "P_uniforme", "P_float32", "Proposta"]


def preparar(seed, args):
    estacoes, eventos = gerar_estacoes(n_estacoes=args.nos, dias=args.dias,
                                       inicio_teste=args.inicio_teste, seed=seed)
    h_teste = args.inicio_teste * 24
    norm = Normalizador().fit([df.iloc[:h_teste] for df in estacoes.values()])
    dados = {}
    for s, df in estacoes.items():
        X, y, t = janelas(df, norm, W=args.W, H=args.H)
        dados[s] = (X, y, t)
    return dados, eventos, norm, h_teste


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
    # erro de referência de cada nó (últimos 14 dias do treino) -> base da severidade
    ref = {}
    modelo.set(theta)
    for s, (X, y, t) in dados.items():
        m = (t >= h_teste - 14 * 24) & (t < h_teste)
        ref[s] = float(np.mean(np.abs(modelo.predict(X[m]) - y[m])))
    return modelo, theta, ref


def simular(estrategia, dados, eventos, h_teste, modelo, theta0, ref, args, seed):
    nos = list(dados.keys())
    local = {s: theta0.copy() for s in nos}          # modelo em uso em cada nó
    theta_g = theta0.copy()                           # modelo global no servidor
    det = {s: PageHinkley(delta=args.ph_delta, lam=args.ph_lambda) for s in nos}
    alerta_ate = {s: -1 for s in nos}
    alarmes = {s: [] for s in nos}
    bits = 32 if estrategia in ("B3_fedavg", "P_float32") else args.bits
    bytes_up = bytes_down = 0
    registros = []                                    # (nó, hora, erro)
    n_dias = (len(next(iter(dados.values()))[2]) + args.W + args.H) // 24 - h_teste // 24

    for d in range(n_dias):
        h_ini, h_fim = h_teste + d * 24, h_teste + (d + 1) * 24
        buffer = {}
        # 1) previsão prequencial do dia (prevê, depois observa) + detector de drift
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
            buffer[s] = (X[mb], y[mb], float(np.mean(np.abs(err[-24:]))))
        if not buffer:
            break

        # 2) atualização conforme a estratégia
        if estrategia == "B1_estatico":
            continue

        if estrategia == "B2_local":
            for s, (Xb, yb, _) in buffer.items():
                local[s] = local[s] + treino_local(modelo, local[s], Xb, yb, epochs=args.ep, lr=args.lr, seed=seed + d)
            continue

        if estrategia == "B3_fedavg":
            deltas = [treino_local(modelo, theta_g, Xb, yb, epochs=args.ep, lr=args.lr, seed=seed + d)
                      for (Xb, yb, _) in buffer.values()]
            n = modelo.n_params
            bytes_up += len(deltas) * n * 4
            bytes_down += len(nos) * n * 4
            theta_g = agregar(theta_g, deltas, [1] * len(deltas))
            for s in nos:
                local[s] = theta_g.copy()
            continue

        # Propostas: só participa quem está em alerta de drift
        part = [s for s in buffer if d <= alerta_ate[s]]
        if not part:
            continue
        deltas, pesos = [], []
        for s in part:
            Xb, yb, err_dia = buffer[s]
            delta = treino_local(modelo, theta_g, Xb, yb, epochs=args.ep, lr=args.lr, seed=seed + d)
            delta_q, nb = quantizar(delta, bits)
            deltas.append(delta_q); bytes_up += nb
            sev = err_dia / ref[s]
            pesos.append(1.0 if estrategia == "P_uniforme" else peso_severidade(sev, args.alpha, args.w_max))
        theta_novo = agregar(theta_g, deltas, pesos)
        _, nb_down = quantizar(theta_novo - theta_g, bits)
        theta_g = theta_novo
        # cada nó recebe o modelo global; adota se estiver em alerta ou se o global errar menos
        for s, (Xb, yb, _) in buffer.items():
            bytes_down += nb_down
            if s in part:
                local[s] = theta_g.copy()
            else:
                modelo.set(theta_g); e_g = np.mean(np.abs(modelo.predict(Xb) - yb))
                modelo.set(local[s]); e_l = np.mean(np.abs(modelo.predict(Xb) - yb))
                if e_g < e_l:
                    local[s] = theta_g.copy()

    return registros, alarmes, bytes_up, bytes_down


def metricas(registros, alarmes, eventos, h_teste, sd_temp, b_up, b_down):
    df = pd.DataFrame(registros, columns=["no", "hora", "erro"])
    df["erro_c"] = df["erro"] * sd_temp                      # volta para °C
    em_evento = np.zeros(len(df), bool)
    for s, ev in eventos.items():
        if ev:
            em_evento |= (df["no"].values == s) & (df["hora"].values >= ev[0]) & (df["hora"].values < ev[1])
    nos_normais = [s for s, ev in eventos.items() if ev is None]
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    atrasos, falsos = [], 0
    for s, ev in eventos.items():
        a = np.array(alarmes[s])
        if ev:
            depois = a[a >= ev[0]]
            atrasos.append((depois[0] - ev[0]) if len(depois) else np.nan)
            falsos += int(np.sum(a < ev[0]))
        else:
            falsos += len(a)
    return {
        "rmse_evento_C": rmse(df.loc[em_evento, "erro_c"].values),
        "rmse_nos_normais_C": rmse(df.loc[df["no"].isin(nos_normais), "erro_c"].values),
        "rmse_geral_C": rmse(df["erro_c"].values),
        "atraso_deteccao_h": float(np.nanmedian(atrasos)) if not np.all(np.isnan(atrasos)) else np.nan,
        "falsos_alarmes": falsos,
        "kB_upload": b_up / 1024, "kB_download": b_down / 1024,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--nos", type=int, default=8)
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
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--w_max", type=float, default=4.0)
    ap.add_argument("--dias_alerta", type=int, default=10)
    ap.add_argument("--ph_delta", type=float, default=0.05)
    ap.add_argument("--ph_lambda", type=float, default=8.0)
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    linhas, curvas = [], {}
    for seed in range(args.seeds):
        t0 = time.time()
        dados, eventos, norm, h_teste = preparar(seed, args)
        modelo, theta0, ref = pre_treino(dados, h_teste, args, seed)
        print(f"[seed {seed}] modelo com {modelo.n_params} parâmetros "
              f"({modelo.n_params * 4 / 1024:.1f} kB em float32)")
        for est in ESTRATEGIAS:
            reg, alarmes, bu, bd = simular(est, dados, eventos, h_teste, modelo, theta0, ref, args, seed)
            m = metricas(reg, alarmes, eventos, h_teste, norm.sd[0], bu, bd)
            linhas.append({"seed": seed, "estrategia": est, **m})
            if seed == 0:
                curvas[est] = reg
            print(f"   {est:12s} RMSE evento={m['rmse_evento_C']:.3f} °C  normais={m['rmse_nos_normais_C']:.3f} °C  "
                  f"upload={m['kB_upload']:.0f} kB  atraso={m['atraso_deteccao_h']}")
        print(f"   ({time.time() - t0:.0f} s)")

    res = pd.DataFrame(linhas)
    res.to_csv(f"{args.saida}/resultados.csv", index=False)

    # resumo: média ± IC95% e teste de Wilcoxon pareado (Proposta vs. cada baseline)
    from scipy import stats
    cols = ["rmse_evento_C", "rmse_nos_normais_C", "kB_upload", "atraso_deteccao_h", "falsos_alarmes"]
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
            linha["p_wilcoxon_vs_Proposta"] = f"{stats.wilcoxon(a, b).pvalue:.4f}"
            linha["reducao_rmse_evento_%"] = f"{100 * (b.mean() - a.mean()) / b.mean():.1f}"
        resumo.append(linha)
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(f"{args.saida}/resumo.csv", index=False)
    print("\n=== RESUMO (média ± IC 95%) ===")
    print(resumo.to_string(index=False))
    if args.seeds < 5:
        print("(use --seeds 5 ou mais para o teste de Wilcoxon)")

    # gráfico: RMSE diário nos nós do epicentro (seed 0)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, eventos, norm, h_teste = preparar(0, args)
    epic = [s for s, ev in eventos.items() if ev and ev == min(e for e in eventos.values() if e)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for est, reg in curvas.items():
        df = pd.DataFrame(reg, columns=["no", "hora", "erro"])
        df = df[df.no.isin(epic)]
        df["dia"] = (df.hora - h_teste) // 24
        g = df.groupby("dia")["erro"].apply(lambda e: np.sqrt(np.mean((e * norm.sd[0]) ** 2)))
        ax.plot(g.index, g.values, label=est, lw=2 if est == "Proposta" else 1.2)
    ev = eventos[epic[0]]
    ax.axvspan((ev[0] - h_teste) / 24, (ev[1] - h_teste) / 24, color="orange", alpha=0.15, label="evento extremo")
    ax.set_xlabel("dia do período de teste"); ax.set_ylabel("RMSE diário (°C)")
    ax.set_title("Erro de previsão nos nós do epicentro"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{args.saida}/curva_rmse.png", dpi=150)
    print(f"\nArquivos salvos em {args.saida}/")


if __name__ == "__main__":
    main()
