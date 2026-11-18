"""Modelo TinyML: MLP pequena (1 camada oculta) implementada em NumPy.

Pequena o bastante para caber num ESP32 (≈1.300 parâmetros). Os pesos ficam
num único vetor, o que simplifica o aprendizado federado e a quantização.
Para o artigo, dá para trocar por PyTorch/TensorFlow e exportar com TFLite Micro.
"""
import numpy as np


class MLP:
    def __init__(self, n_in, n_hid=32, seed=0):
        rng = np.random.default_rng(seed)
        self.shapes = [(n_in, n_hid), (n_hid,), (n_hid, 1), (1,)]
        w1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hid))
        w2 = rng.normal(0, np.sqrt(1 / n_hid), (n_hid, 1))
        self.theta = np.concatenate([w1.ravel(), np.zeros(n_hid), w2.ravel(), np.zeros(1)]).astype(np.float32)

    # ---- acesso aos pesos ----
    def _split(self, th):
        out, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(th[i:i + n].reshape(s))
            i += n
        return out

    def get(self):
        return self.theta.copy()

    def set(self, th):
        self.theta = th.astype(np.float32).copy()

    @property
    def n_params(self):
        return self.theta.size

    # ---- inferência e treino ----
    def predict(self, X):
        w1, b1, w2, b2 = self._split(self.theta)
        h = np.maximum(0, X @ w1 + b1)
        return (h @ w2 + b2).ravel()

    def fit(self, X, y, epochs=1, lr=1e-3, batch=64, seed=0, so_ultima_camada=False):
        """Adam simples. so_ultima_camada=True imita o ajuste leve feito no dispositivo."""
        rng = np.random.default_rng(seed)
        m = np.zeros_like(self.theta); v = np.zeros_like(self.theta)
        b1_, b2_, eps, step = 0.9, 0.999, 1e-8, 0
        n_w1 = self.shapes[0][0] * self.shapes[0][1] + self.shapes[1][0]
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                j = order[i:i + batch]
                g = self._grad(X[j], y[j])
                if so_ultima_camada:
                    g[:n_w1] = 0
                step += 1
                m = b1_ * m + (1 - b1_) * g
                v = b2_ * v + (1 - b2_) * g * g
                mh = m / (1 - b1_ ** step); vh = v / (1 - b2_ ** step)
                self.theta -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def _grad(self, X, y):
        w1, b1, w2, b2 = self._split(self.theta)
        a = X @ w1 + b1
        h = np.maximum(0, a)
        pred = (h @ w2 + b2).ravel()
        d = (2 * (pred - y) / len(y))[:, None]          # derivada do MSE
        gw2 = h.T @ d; gb2 = d.sum(0)
        dh = (d @ w2.T) * (a > 0)
        gw1 = X.T @ dh; gb1 = dh.sum(0)
        return np.concatenate([gw1.ravel(), gb1, gw2.ravel(), gb2]).astype(np.float32)
