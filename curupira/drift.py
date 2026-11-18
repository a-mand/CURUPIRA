"""Detector de concept drift: Page-Hinkley sobre o erro absoluto da previsão.

Leve o bastante para rodar no ESP32 (guarda só 4 números). Se a biblioteca
River estiver instalada, dá para comparar com ADWIN e KSWIN (ver README).
"""


class PageHinkley:
    """Dispara quando o erro médio sobe de forma persistente.

    delta: tolerância a variações pequenas; lam: limiar de alarme;
    alpha: esquecimento (1.0 = sem esquecimento).
    """

    def __init__(self, delta=0.05, lam=8.0, alpha=0.999, min_amostras=48):
        self.delta, self.lam, self.alpha, self.min = delta, lam, alpha, min_amostras
        self.reset()

    def reset(self):
        self.n, self.media, self.m, self.M = 0, 0.0, 0.0, 0.0

    def update(self, x):
        """Recebe um novo erro absoluto. Retorna True se detectou drift."""
        self.n += 1
        self.media += (x - self.media) / self.n
        self.m = self.alpha * self.m + (x - self.media - self.delta)
        self.M = min(self.M, self.m)
        if self.n >= self.min and (self.m - self.M) > self.lam:
            self.reset()
            return True
        return False


def detector_river(nome="ADWIN"):
    """Opcional: detectores da biblioteca River (pip install river)."""
    from river import drift
    return {"ADWIN": drift.ADWIN, "KSWIN": drift.KSWIN, "PH": drift.PageHinkley}[nome]()
