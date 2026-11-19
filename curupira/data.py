"""Dados: gerador sintético com evento extremo (para testar o pipeline) e leitor de CSV do INMET.

Cada estação vira um "nó", e os nós são agrupados por microrregião.
Os dados são séries horárias com 3 variáveis:
temperatura (°C), umidade relativa (%) e pressão (hPa).
"""
import numpy as np
import pandas as pd

FEATURES = ["temp", "umid", "pressao"]


# ---------------------------------------------------------------------------
# 1) Gerador sintético por microrregiões (substituir por dados reais do INMET)
# ---------------------------------------------------------------------------
# Microrregiões geográficas do Pará (IBGE, 1990) usadas como exemplo ilustrativo.
# O papel de cada uma no evento (epicentro / vizinha / fora) é HIPOTÉTICO e só serve
# para testar o código; com dados reais, os papéis saem da rotulagem dos eventos.
MICRORREGIOES = {
    "Santarém": "epicentro",
    "Óbidos": "epicentro",
    "Itaituba": "vizinha",
    "Altamira": "vizinha",
    "Marabá": "fora",
    "Belém": "fora",
}


def gerar_microrregioes(estacoes_por_mr=4, dias=420, inicio_teste=300, seed=0,
                        microrregioes=MICRORREGIOES, dia_evento=40, atraso_vizinhas=10, duracao=45):
    """Gera séries horárias de clima para várias estações agrupadas em microrregiões.

    Estações da mesma microrregião compartilham o clima regional (com variação local).
    Uma 'seca extrema' atinge primeiro as microrregiões do epicentro e, dias depois,
    as vizinhas, com menor intensidade. O evento é um concept drift real: além de subir a
    temperatura e baixar a umidade, ele muda a DINÂMICA (amplitude diurna maior e maior
    persistência do calor).

    Retorna:
      estacoes  {id: DataFrame}
      eventos   {id: (h_inicio, h_fim) ou None}
      mr_de     {id: nome da microrregião}
      papel     {nome da microrregião: 'epicentro' | 'vizinha' | 'fora'}
    """
    rng = np.random.default_rng(seed)
    H = dias * 24
    t = np.arange(H)
    hora = t % 24
    dia = t // 24
    estacoes, eventos, mr_de = {}, {}, {}
    sid = 0
    for nome, papel in microrregioes.items():
        base_mr = 27 + rng.normal(0, 0.8)          # clima base da microrregião
        amp_mr = 4 + rng.normal(0, 0.4)            # amplitude diurna regional
        fase_mr = rng.uniform(0, 1)
        for _ in range(estacoes_por_mr):
            base_t = base_mr + rng.normal(0, 0.3)  # variação local da estação
            amp = amp_mr + rng.normal(0, 0.15)
            sazonal = 1.2 * np.sin(2 * np.pi * dia / 365 + fase_mr)
            diurno = np.sin(2 * np.pi * (hora - 9) / 24)

            k = np.zeros(H)
            ev = None
            if papel in ("epicentro", "vizinha"):
                d0 = inicio_teste + dia_evento + (atraso_vizinhas if papel == "vizinha" else 0)
                d0 += int(rng.integers(-1, 2))     # estações da mesma MR não começam no mesmo dia
                intens = (1.0 if papel == "epicentro" else 0.6) * rng.uniform(0.9, 1.1)
                h0, h1 = d0 * 24, (d0 + duracao) * 24
                ramp = 7 * 24
                k[h0:h1] = intens * np.minimum(1, (t[h0:h1] - h0) / ramp)
                k[h1 - ramp:h1] = np.minimum(k[h1 - ramp:h1], intens * (h1 - t[h1 - ramp:h1]) / ramp)
                ev = (h0, h1)

            ruido = np.zeros(H)
            e = rng.normal(0, 0.6, H)
            for i in range(1, H):
                phi = 0.85 + 0.12 * k[i]
                ruido[i] = phi * ruido[i - 1] + e[i]

            temp = base_t + sazonal + amp * (1 + 0.8 * k) * diurno + 4.0 * k + ruido
            umid = np.clip(80 - 2.2 * (temp - base_t) - 25 * k + rng.normal(0, 3, H), 15, 100)
            press = 1010 + 1.5 * np.sin(2 * np.pi * (hora - 10) / 12) - 2.0 * k + rng.normal(0, 0.5, H)
            estacoes[sid] = pd.DataFrame({"temp": temp, "umid": umid, "pressao": press, "hora": hora})
            eventos[sid] = ev
            mr_de[sid] = nome
            sid += 1
    return estacoes, eventos, mr_de, dict(microrregioes)


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
