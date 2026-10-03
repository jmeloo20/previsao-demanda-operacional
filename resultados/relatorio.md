# Demanda: resultados e decisão

O modelo selecionado nos três backtests foi **ridge_10**. No teste de 01/10 a 31/12/2012, apresentou **MAE 960.8 aluguéis/dia**, RMSE 1355.6 e WAPE 18.6%. A referência sazonal, que repete a demanda de sete dias antes, teve MAE 1457.6. A redução relativa de MAE foi **34.1%**.

O IC exploratório de 95% da diferença de MAE (modelo menos referência) foi [-867.5, -248.9], calculado por bootstrap em blocos móveis de sete dias. Valores negativos favorecem o modelo; intervalo cruzando zero indica incerteza na vantagem. Esse intervalo é condicionado aos modelos treinados, à janela observada e à escolha do bloco.

O intervalo empírico calibrado em agosto e setembro teve cobertura de **81.5%** no teste, contra 90% nominal. A largura média foi 3157.3 aluguéis. Dependência temporal e mudança de distribuição impedem promessa de cobertura futura.

## Decisão apoiada

Antecipar o volume agregado do próximo dia pode apoiar planejamento de atendimento e redistribuição. Aluguéis por dia não equivalem ao tamanho necessário da frota: faltam duração das viagens, estações, estoque e demanda reprimida. Não foi estimado número de bicicletas, funcionários nem economia financeira.

## Como a avaliação funciona

A cada dia de teste, a demanda até a véspera é considerada conhecida. Os coeficientes ficam congelados após o treino de julho. Isso simula 92 previsões sucessivas de um dia, não uma previsão única de 92 dias. Médias móveis são deslocadas antes do cálculo. casual e registered somam o alvo e são excluídos; clima realizado do dia previsto também é excluído porque previsões meteorológicas históricas não estão disponíveis.

## Limitações e continuidade

Dados de uma cidade nos Estados Unidos em 2011–2012 não representam diretamente uma operação brasileira atual. Há somente dois anos e alguns dias de erro elevado. Não se atribui a causa desses erros sem dados externos. Os próximos passos seriam validar novas temporadas, incorporar previsões de clima disponíveis no momento da decisão e modelar estações com dados de capacidade.
