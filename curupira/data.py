"""Dados: gerador sintético com evento extremo (para testar o pipeline) e leitor de CSV do INMET.

Cada estação vira um "nó". Os dados são séries horárias com 3 variáveis:
temperatura (°C), umidade relativa (%) e pressão (hPa).
"""
import numpy as np
import pandas as pd

FEATURES = ["temp", "umid", "pressao"]


# ---------------------------------------------------------------------------
# 1) Gerador sintético (substituir por dados reais do INMET quando disponíveis)
# ---------------------------------------------------------------------------
def gerar_estacoes(n_estacoes=8, dias=420, inicio_teste=300, seed=0,
                   epicentro=(0, 1, 2), vizinhas=(3, 4), dia_evento=40, atraso_vizinhas=10,
                   duracao=45):
    """Gera séries horárias de clima com uma 'seca extrema' que atinge
    primeiro as estações do epicentro e, depois, as vizinhas.

    O evento é um concept drift real: além de subir a média de temperatura e
    baixar a umidade, ele muda a DINÂMICA (amplitude diurna maior e maior
    persistência do calor), de modo que a relação entrada -> saída aprendida
    em regime normal deixa de valer.
    Retorna: dict {id_estacao: DataFrame}, dict {id_estacao: (h_inicio, h_fim) do evento ou None}
    """
    rng = np.random.default_rng(seed)
    H = dias * 24
    t = np.arange(H)
    hora = t % 24
    dia = t // 24
    estacoes, eventos = {}, {}
    for s in range(n_estacoes):
        base_t = 27 + rng.normal(0, 0.8)           # clima base da estação
        amp = 4 + rng.normal(0, 0.4)               # amplitude diurna
        sazonal = 1.2 * np.sin(2 * np.pi * dia / 365 + rng.uniform(0, 1))
        diurno = np.sin(2 * np.pi * (hora - 9) / 24)

        # intensidade do evento ao longo do tempo (rampa de subida e descida)
        k = np.zeros(H)
        ev = None
        if s in epicentro or s in vizinhas:
            d0 = inicio_teste + dia_evento + (atraso_vizinhas if s in vizinhas else 0)
            intens = 1.0 if s in epicentro else 0.6
            h0, h1 = d0 * 24, (d0 + duracao) * 24
            ramp = 7 * 24
            k[h0:h1] = intens * np.minimum(1, (t[h0:h1] - h0) / ramp)
            k[h1 - ramp:h1] = np.minimum(k[h1 - ramp:h1], intens * (h1 - t[h1 - ramp:h1]) / ramp)
            ev = (h0, h1)
        eventos[s] = ev

        # ruído autocorrelacionado (AR(1)); persistência maior durante o evento
        ruido = np.zeros(H)
        e = rng.normal(0, 0.6, H)
        for i in range(1, H):
            phi = 0.85 + 0.12 * k[i]
            ruido[i] = phi * ruido[i - 1] + e[i]

        temp = base_t + sazonal + amp * (1 + 0.8 * k) * diurno + 4.0 * k + ruido
        umid = np.clip(80 - 2.2 * (temp - base_t) - 25 * k + rng.normal(0, 3, H), 15, 100)
        press = 1010 + 1.5 * np.sin(2 * np.pi * (hora - 10) / 12) - 2.0 * k + rng.normal(0, 0.5, H)
        estacoes[s] = pd.DataFrame({"temp": temp, "umid": umid, "pressao": press, "hora": hora})
    return estacoes, eventos


# ---------------------------------------------------------------------------
# 2) Leitor de dados reais do INMET (BDMEP / dados históricos anuais)
# ---------------------------------------------------------------------------
def ler_inmet_csv(caminho):
    """Lê um CSV de estação automática do INMET (arquivo anual do portal de dados históricos).

    Formato típico: separador ';', decimal ',', codificação latin-1 e 8 linhas de
    cabeçalho com metadados. Os nomes das colunas mudam um pouco entre anos,
    por isso a busca é por trechos do nome. Confira com um arquivo real.
    """
    df = pd.read_csv(caminho, sep=";", decimal=",", encoding="latin-1", skiprows=8)
    def col(trecho):
        for c in df.columns:
            if trecho.lower() in c.lower():
                return c
        raise KeyError(trecho)
    out = pd.DataFrame({
        "temp": pd.to_numeric(df[col("TEMPERATURA DO AR - BULBO SECO")], errors="coerce"),
        "umid": pd.to_numeric(df[col("UMIDADE RELATIVA DO AR, HORARIA")], errors="coerce"),
        "pressao": pd.to_numeric(df[col("PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA")], errors="coerce"),
    })
    out[out <= -9999] = np.nan                       # INMET usa -9999 para falha
    out = out.interpolate(limit=6).ffill().bfill()   # preenche lacunas curtas
    hora_col = col("Hora")
    out["hora"] = df[hora_col].astype(str).str.extract(r"(\d{2})")[0].astype(int)
    return out


# ---------------------------------------------------------------------------
# 3) Janelas deslizantes: últimas W horas -> temperatura daqui a H horas
# ---------------------------------------------------------------------------
class Normalizador:
    def fit(self, dfs):
        allv = np.concatenate([d[FEATURES].values for d in dfs])
        self.mu, self.sd = allv.mean(0), allv.std(0) + 1e-6
        return self

    def transform(self, df):
        return (df[FEATURES].values - self.mu) / self.sd


def janelas(df, norm, W=12, H=3):
    """Monta X (W horas x 3 variáveis + seno/cosseno da hora) e y (temp. em t+H, normalizada)."""
    z = norm.transform(df)
    hora = df["hora"].values
    n = len(df) - W - H + 1
    idx = np.arange(W)[None, :] + np.arange(n)[:, None]
    Xw = z[idx].reshape(n, -1)
    h_alvo = hora[W - 1 + H: W - 1 + H + n]
    Xh = np.stack([np.sin(2 * np.pi * h_alvo / 24), np.cos(2 * np.pi * h_alvo / 24)], 1)
    X = np.hstack([Xw, Xh]).astype(np.float32)
    y = z[W - 1 + H: W - 1 + H + n, 0].astype(np.float32)
    t_alvo = np.arange(W - 1 + H, W - 1 + H + n)   # índice horário do valor previsto
    return X, y, t_alvo
