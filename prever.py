import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelagem import preparar, prever

RAIZ = Path(__file__).resolve().parent


def executar():
    parser = argparse.ArgumentParser(description="Prever demanda a partir de atributos calculados até a véspera.")
    parser.add_argument("entrada", type=Path)
    parser.add_argument("saida", type=Path)
    parser.add_argument("--historico", action="store_true")
    parser.add_argument("--feriado", type=int, choices=[0, 1])
    parser.add_argument("--dia-util", type=int, choices=[0, 1])
    args = parser.parse_args()
    pacote = joblib.load(RAIZ / "resultados/modelo.joblib")
    if args.historico:
        if args.feriado is None or args.dia_util is None:
            raise ValueError("Informe --feriado e --dia-util para a próxima data.")
        historico = pd.read_csv(args.entrada)
        proxima = pd.to_datetime(historico.dteday).max() + pd.Timedelta(days=1)
        linha = pd.DataFrame({"dteday": [str(proxima.date())], "cnt": [np.nan], "holiday": [args.feriado], "workingday": [args.dia_util]})
        x, _ = preparar(pd.concat([historico, linha], ignore_index=True), permitir_ultimo_sem_alvo=True)
        x = x.loc[[proxima]]
    else:
        x = pd.read_csv(args.entrada, index_col="data", parse_dates=True)
    if set(x.columns) != set(pacote["colunas"]):
        raise ValueError(f"Colunas esperadas: {pacote['colunas']}")
    x = x[pacote["colunas"]].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Entrada contém dados ausentes ou infinitos.")
    p = prever(pacote["modelo"], x)
    pd.DataFrame({"previsto": p, "inferior_90": np.maximum(0, p - pacote["raio"]), "superior_90": p + pacote["raio"]}, index=x.index).to_csv(args.saida)
    print(f"{len(x)} previsões salvas; cada linha exige histórico real disponível até a véspera.")


if __name__ == "__main__":
    executar()
