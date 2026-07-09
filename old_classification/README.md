# old_classification/ — código aposentado (NÃO USAR)

Este diretório guarda o código da **hipótese antiga** do projeto: **classificação do
*grading* de retinopatia diabética** (5 classes: No DR → Proliferative DR), com modelos
tipo MobileNet e métricas de classificação (accuracy, precision/recall/F1 macro, AUC-ROC
macro OvR, matriz de confusão).

**Essa hipótese foi descontinuada.** O trabalho atual é a nova hipótese descrita em
[`../docs/Especificacao_Implementacao_YOLO_SAHI_DDR.md`](../docs/Especificacao_Implementacao_YOLO_SAHI_DDR.md):
**detecção de lesões (YOLO vs. YOLO + SAHI)** no dataset DDR, com métricas de detecção (mAP).

Mantido apenas como referência histórica. Não é ponto de partida da implementação nova e
não deve ser importado/executado pelo código atual. O dataset continua fora daqui, em
`../data/` (não foi movido).
