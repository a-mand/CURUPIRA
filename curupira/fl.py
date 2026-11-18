"""Aprendizado federado: agregação (uniforme ou ponderada por severidade) e quantização.

Tudo é simulado no mesmo computador: cada "cliente" só enxerga os próprios dados.
Para usar o framework Flower, basta mover `treino_local` para um flwr.client.NumPyClient
e `agregar` para uma Strategy personalizada (ver README).
"""
import numpy as np


# ------------------------------- quantização --------------------------------
def quantizar(delta, bits=8):
    """Quantização simétrica por tensor: float32 -> inteiros de `bits` bits + 1 escala float32.
    Retorna (vetor reconstruído, bytes que seriam transmitidos)."""
    if bits >= 32:
        return delta.copy(), delta.size * 4
    qmax = 2 ** (bits - 1) - 1
    escala = float(np.abs(delta).max()) / qmax if np.any(delta) else 1.0
    q = np.clip(np.round(delta / escala), -qmax, qmax)
    n_bytes = int(np.ceil(delta.size * bits / 8)) + 4
    return (q * escala).astype(np.float32), n_bytes


# -------------------------------- agregação ----------------------------------
def agregar(theta_global, deltas, pesos):
    """Média ponderada das atualizações (deltas) dos clientes."""
    pesos = np.asarray(pesos, dtype=np.float64)
    pesos = pesos / pesos.sum()
    return theta_global + sum(p * d for p, d in zip(pesos, deltas)).astype(np.float32)


def peso_severidade(severidade, alpha=2.0, w_max=4.0):
    """Peso do cliente na agregação: cresce com a severidade do evento,
    com teto w_max para não 'atropelar' os nós fora do evento."""
    return float(min(w_max, 1.0 + alpha * max(0.0, severidade - 1.0)))


def treino_local(modelo, theta_global, X, y, epochs=3, lr=2e-3, seed=0, so_ultima_camada=False):
    """O cliente parte do modelo global, treina nos dados locais e devolve o delta."""
    modelo.set(theta_global)
    modelo.fit(X, y, epochs=epochs, lr=lr, seed=seed, so_ultima_camada=so_ultima_camada)
    return modelo.get() - theta_global
