import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dados import verificar
from modelagem import intervalo_diferenca, metricas, preparar, prever, raio_empirico

RAIZ = Path(__file__).resolve().parent


def executar():
    verificar()
    original = pd.read_csv(RAIZ / "dados/day.csv")
    x, y = preparar(original)
    saida = RAIZ / "resultados"
    saida.mkdir(exist_ok=True)
    candidatos = {"sazonal_7_dias": None, "ridge_10": make_pipeline(StandardScaler(), Ridge(alpha=10)), "ridge_100": make_pipeline(StandardScaler(), Ridge(alpha=100))}
    for folhas in [7, 15]:
        candidatos[f"boosting_{folhas}"] = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=folhas, learning_rate=.05, l2_regularization=10, early_stopping=False, random_state=42)
    janelas = [("2012-04-30", "2012-05-01", "2012-05-31"), ("2012-05-31", "2012-06-01", "2012-06-30"), ("2012-06-30", "2012-07-01", "2012-07-31")]
    registros = []
    for corte, inicio, fim in janelas:
        treino = x.index <= corte
        validacao = (x.index >= inicio) & (x.index <= fim)
        for nome, estimador in candidatos.items():
            modelo = None if estimador is None else clone(estimador).fit(x.loc[treino], y.loc[treino])
            p = prever(modelo, x.loc[validacao])
            registros.append({"modelo": nome, "treino_ate": corte, "validacao_inicio": inicio, "validacao_fim": fim, **metricas(y.loc[validacao], p)})
    backtest = pd.DataFrame(registros)
    backtest.to_csv(saida / "backtests.csv", index=False)
    medias = backtest.groupby("modelo").mae.mean().sort_values()
    escolhido = medias.index[0]
    selecao = {"modelo": escolhido, "criterio": "Menor MAE médio nos três backtests mensais", "horizonte_dias": 1, "treino_ate": "2012-07-31", "calibracao": ["2012-08-01", "2012-09-30"], "teste": ["2012-10-01", "2012-12-31"], "uso_historico": "Demanda real até o dia anterior disponível em cada previsão diária"}
    (saida / "selecao.json").write_text(json.dumps(selecao, indent=2, ensure_ascii=False))
    treino = x.index <= "2012-07-31"
    calibracao = (x.index >= "2012-08-01") & (x.index <= "2012-09-30")
    teste = x.index >= "2012-10-01"
    modelos, avaliacoes = {}, []
    for nome, estimador in candidatos.items():
        modelo = None if estimador is None else clone(estimador).fit(x.loc[treino], y.loc[treino])
        modelos[nome] = modelo
        avaliacoes.append({"modelo": nome, "selecionado": nome == escolhido, **metricas(y.loc[teste], prever(modelo, x.loc[teste]))})
    pd.DataFrame(avaliacoes).to_csv(saida / "avaliacao_teste.csv", index=False)
    modelo = modelos[escolhido]
    raio = raio_empirico(np.abs(y.loc[calibracao] - prever(modelo, x.loc[calibracao])))
    pred = prever(modelo, x.loc[teste])
    sazonal = prever(None, x.loc[teste])
    resultados = pd.DataFrame({"real": y.loc[teste], "previsto": pred, "sazonal": sazonal, "inferior_90": np.maximum(0, pred - raio), "superior_90": pred + raio})
    resultados["coberto"] = resultados.real.between(resultados.inferior_90, resultados.superior_90)
    resultados["erro_absoluto"] = abs(resultados.real - resultados.previsto)
    resultados.to_csv(saida / "previsoes_teste.csv")
    resultados.sort_values("erro_absoluto", ascending=False).head(10).to_csv(saida / "maiores_erros.csv")
    resultados.groupby(resultados.index.to_period("M")).agg(dias=("real", "size"), mae=("erro_absoluto", "mean"), cobertura=("coberto", "mean")).to_csv(saida / "erros_mensais.csv")
    x.assign(demanda=y, particao=np.select([treino, calibracao, teste], ["treino_final", "calibracao", "teste"], default="nao_usado")).to_csv(saida / "dados_modelagem.csv")
    teste_metricas = metricas(y.loc[teste], pred)
    referencia = metricas(y.loc[teste], sazonal)
    ic = intervalo_diferenca(y.loc[teste], pred, sazonal)
    reducao = 1 - teste_metricas["mae"] / referencia["mae"]
    resumo = {"dias_originais": len(original), "dias_modelagem": len(x), "dias_treino": int(treino.sum()), "dias_calibracao": int(calibracao.sum()), "dias_teste": int(teste.sum()), "modelo": escolhido, "metricas_teste": teste_metricas, "metricas_sazonal": referencia, "reducao_mae_vs_sazonal": reducao, "ic95_diferenca_mae_blocos": ic.tolist(), "raio_intervalo": raio, "cobertura_teste": float(resultados.coberto.mean()), "largura_media_intervalo": float((resultados.superior_90 - resultados.inferior_90).mean()), "semente": 42}
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    joblib.dump({"modelo": modelo, "colunas": list(x.columns), "raio": raio}, saida / "modelo.joblib")
    x.loc[teste].head().to_csv(RAIZ / "exemplo_entrada.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, eixo = plt.subplots(figsize=(11, 4.5))
    eixo.fill_between(resultados.index, resultados.inferior_90, resultados.superior_90, color="#236e94", alpha=.15, label="Intervalo empírico, nominal 90%")
    eixo.plot(resultados.index, resultados.real, color="#202b3c", label="Observado", lw=1.5)
    eixo.plot(resultados.index, resultados.previsto, color="#287faa", label="Previsão de um dia", lw=1.3)
    eixo.set(ylabel="Aluguéis diários", title="Demanda operacional: teste cronológico reservado")
    eixo.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(saida / "previsao.png", dpi=180)
    plt.close(fig)
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    medias.plot.barh(ax=eixos[0], color="#287faa")
    eixos[0].set(xlabel="MAE médio", ylabel="", title="Seleção nos backtests")
    eixos[1].scatter(resultados.previsto, resultados.real - resultados.previsto, s=20, alpha=.7, color="#287faa")
    eixos[1].axhline(0, color="#555555", lw=1)
    eixos[1].set(xlabel="Previsão", ylabel="Observado − previsto", title="Resíduos do teste")
    fig.tight_layout()
    fig.savefig(saida / "diagnostico.png", dpi=180)
    plt.close(fig)
    texto = f'''# Demanda: resultados e decisão

O modelo selecionado nos três backtests foi **{escolhido}**. No teste de 01/10 a 31/12/2012, apresentou **MAE {teste_metricas['mae']:.1f} aluguéis/dia**, RMSE {teste_metricas['rmse']:.1f} e WAPE {teste_metricas['wape']:.1%}. A referência sazonal, que repete a demanda de sete dias antes, teve MAE {referencia['mae']:.1f}. A redução relativa de MAE foi **{reducao:.1%}**.

O IC exploratório de 95% da diferença de MAE (modelo menos referência) foi [{ic[0]:.1f}, {ic[1]:.1f}], calculado por bootstrap em blocos móveis de sete dias. Valores negativos favorecem o modelo; intervalo cruzando zero indica incerteza na vantagem. Esse intervalo é condicionado aos modelos treinados, à janela observada e à escolha do bloco.

O intervalo empírico calibrado em agosto e setembro teve cobertura de **{resultados.coberto.mean():.1%}** no teste, contra 90% nominal. A largura média foi {resumo['largura_media_intervalo']:.1f} aluguéis. Dependência temporal e mudança de distribuição impedem promessa de cobertura futura.

## Decisão apoiada

Antecipar o volume agregado do próximo dia pode apoiar planejamento de atendimento e redistribuição. Aluguéis por dia não equivalem ao tamanho necessário da frota: faltam duração das viagens, estações, estoque e demanda reprimida. Não foi estimado número de bicicletas, funcionários nem economia financeira.

## Como a avaliação funciona

A cada dia de teste, a demanda até a véspera é considerada conhecida. Os coeficientes ficam congelados após o treino de julho. Isso simula 92 previsões sucessivas de um dia, não uma previsão única de 92 dias. Médias móveis são deslocadas antes do cálculo. casual e registered somam o alvo e são excluídos; clima realizado do dia previsto também é excluído porque previsões meteorológicas históricas não estão disponíveis.

## Limitações e continuidade

Dados de uma cidade nos Estados Unidos em 2011–2012 não representam diretamente uma operação brasileira atual. Há somente dois anos e alguns dias de erro elevado. Não se atribui a causa desses erros sem dados externos. Os próximos passos seriam validar novas temporadas, incorporar previsões de clima disponíveis no momento da decisão e modelar estações com dados de capacidade.
'''
    (saida / "relatorio.md").write_text(texto, encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    executar()
