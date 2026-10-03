import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def preparar(original, permitir_ultimo_sem_alvo=False):
    dados = original.copy()
    dados["data"] = pd.to_datetime(dados.dteday)
    dados = dados.sort_values("data").set_index("data")
    ausentes = dados.cnt.isna()
    futuro_valido = permitir_ultimo_sem_alvo and ausentes.sum() == 1 and ausentes.iloc[-1]
    if dados.index.has_duplicates or (ausentes.any() and not futuro_valido) or (dados.cnt < 0).any():
        raise ValueError("Datas duplicadas ou demanda inválida.")
    if not dados.index.equals(pd.date_range(dados.index.min(), dados.index.max(), freq="D", name="data")):
        raise ValueError("Calendário tem lacunas; não preencher demanda ausente com zero.")
    x = pd.DataFrame(index=dados.index)
    x["feriado"] = dados.holiday.astype(int)
    x["dia_util"] = dados.workingday.astype(int)
    x["dia_semana_seno"] = np.sin(2 * np.pi * dados.index.dayofweek / 7)
    x["dia_semana_cosseno"] = np.cos(2 * np.pi * dados.index.dayofweek / 7)
    x["ano_seno"] = np.sin(2 * np.pi * dados.index.dayofyear / 365.25)
    x["ano_cosseno"] = np.cos(2 * np.pi * dados.index.dayofyear / 365.25)
    x["tendencia_dias"] = (dados.index - pd.Timestamp("2011-01-01")).days
    for atraso in [1, 7, 14, 28]:
        x[f"lag_{atraso}"] = dados.cnt.shift(atraso)
    for janela in [7, 28]:
        x[f"media_{janela}"] = dados.cnt.shift(1).rolling(janela).mean()
    x["desvio_7"] = dados.cnt.shift(1).rolling(7).std()
    x = x.dropna()
    return x, dados.loc[x.index, "cnt"].astype(float)


def metricas(real, previsto):
    real, previsto = np.asarray(real), np.asarray(previsto)
    return {"mae": float(mean_absolute_error(real, previsto)), "rmse": float(np.sqrt(mean_squared_error(real, previsto))), "wape": float(np.abs(real - previsto).sum() / real.sum()) if real.sum() else None, "vies": float((previsto - real).mean())}


def prever(modelo, x):
    valores = x.lag_7.to_numpy() if modelo is None else modelo.predict(x)
    return np.maximum(0, valores)


def raio_empirico(erros, nivel=.9):
    erros = np.sort(np.asarray(erros))
    if not len(erros) or not 0 < nivel < 1 or not np.isfinite(erros).all() or (erros < 0).any():
        raise ValueError("Erros ou nível inválidos.")
    ordem = min(len(erros), math.ceil((len(erros) + 1) * nivel))
    return float(erros[ordem - 1])


def intervalo_diferenca(real, previsto, referencia, repeticoes=2000, bloco=7):
    diferencas = np.abs(np.asarray(real) - np.asarray(previsto)) - np.abs(np.asarray(real) - np.asarray(referencia))
    if len(diferencas) < bloco:
        raise ValueError("Série insuficiente para bootstrap em blocos.")
    rng = np.random.default_rng(42)
    medias = []
    for _ in range(repeticoes):
        inicios = rng.integers(0, len(diferencas) - bloco + 1, math.ceil(len(diferencas) / bloco))
        indices = np.concatenate([np.arange(i, i + bloco) for i in inicios])[:len(diferencas)]
        medias.append(diferencas[indices].mean())
    return np.quantile(medias, [.025, .975])
