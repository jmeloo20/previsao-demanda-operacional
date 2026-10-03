import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent


def executar():
    original = pd.read_csv(RAIZ / "dados/day.csv")
    with sqlite3.connect(":memory:") as conexao:
        original.to_sql("demanda", conexao, index=False)
        resultado = pd.read_sql_query((RAIZ / "sql/atributos.sql").read_text(), conexao, parse_dates=["data"])
    python = pd.read_csv(RAIZ / "resultados/dados_modelagem.csv", parse_dates=["data"])
    combinado = python.merge(resultado, on="data", suffixes=("_python", "_sql"), validate="one_to_one")
    if len(combinado) != len(python):
        raise ValueError("Datas divergentes entre Python e SQL.")
    for nome in ["lag_1", "lag_7", "media_7"]:
        if not np.allclose(combinado[f"{nome}_python"], combinado[f"{nome}_sql"]):
            raise ValueError(f"Divergência em {nome}.")
    resultado.to_csv(RAIZ / "resultados/atributos_sql.csv", index=False)
    print(f"{len(combinado)} datas verificadas: lags e média móvel conferem em SQL e Python.")


if __name__ == "__main__":
    executar()
