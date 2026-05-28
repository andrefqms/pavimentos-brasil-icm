# Estudo de Caso — Detecção de Patologias em Pavimentos Urbanos Brasileiros com YOLOv8

**Data:** 28 de maio de 2026
**Modelo base:** YOLOv8m
**Dataset:** Pavimentos-Brasil

---

## 1. Contexto e Motivação

A deterioração de pavimentos urbanos é um problema recorrente nas cidades brasileiras, impactando diretamente a segurança viária, o custo de manutenção e a qualidade de vida da população. Inspeções manuais são caras, lentas e sujeitas a inconsistências entre avaliadores.

Este estudo aplica visão computacional com YOLOv8 para automatizar a detecção de patologias e calcular o **Índice de Condição da Manutenção (ICM)**, fornecendo uma métrica objetiva e reprodutível para priorização de reparos.

---

## 2. Dataset

### 2.1 Composição

Autor: Mateus Serafim
Disponivel em: https://www.kaggle.com/datasets/mateusserafim/pavimentosbrasil
Licenca: ver pagina do dataset no Kaggle
Conteudo: aproximadamente 9.000 imagens JPEG de rodovias do Ceara e Piaui, organizadas por rodovia.

O dataset **Pavimentos-Brasil** é composto por imagens coletadas em vias públicas de municípios brasileiros, anotadas no formato YOLO (bounding boxes normalizadas). 

| Atributo           | Valor                    |
|--------------------|--------------------------|
| Formato            | YOLO (`.txt` por imagem) |
| Tamanho de entrada | 640 × 640 px             |
| Splits             | 70% treino / 15% val / 15% teste |

### 2.2 Classes e Distribuição de Pesos ICM

| Classe                   | Peso ICM | Justificativa                                           |
|--------------------------|----------|---------------------------------------------------------|
| `buraco`                 | 5        | Risco imediato de dano a veículos e pedestres           |
| `trinca`                 | 3        | Indicativo de falha estrutural progressiva              |
| `remendo`                | 2        | Reparo temporário; sinaliza histórico de deterioração   |
| `drenagem`               | 2        | Escoamento inadequado acelera degradação                |
| `sinalizacao_vertical`   | 1        | Influência indireta na segurança                        |
| `sinalizacao_horizontal` | 1        | Influência indireta na segurança                        |
| `vegetacao`              | 1        | Dano potencial a longo prazo                            |

---

### 2.3 Limitações Conhecidas

Este projeto utiliza pseudo-labels gerados automaticamente pelo GroundingDINO, o que pode introduzir falsas detecções e anotações imprecisas. Além disso, o modelo apresenta limitações na detecção de defeitos pequenos, como trincas finas e desgastes superficiais. O cálculo do ICM implementado neste trabalho é uma estimativa simplificada baseada na área detectada dos danos e não substitui avaliações técnicas oficiais realizadas por especialistas.

## 3. Metodologia

### 3.1 Pré-processamento

O script `02_preprocessing.py` executa as seguintes etapas:

1. Validação de pares imagem/anotação
2. Divisão estratificada por classe (70/15/15)
3. Geração da estrutura de diretórios compatível com Ultralytics
4. Criação do arquivo `dataset.yaml`

### 3.2 Treinamento

```
Modelo base  : yolov8m.pt (pré-treinado em COCO)
Épocas       : 100
Batch size   : 16
Img size     : 640
Otimizador   : AdamW (padrão Ultralytics)
Augmentation : Mosaic, Mixup, HSV, Flip horizontal
```

O treinamento foi executado com `patience=20` (early stopping), salvando o melhor checkpoint em `models/best.pt`.

### 3.3 Cálculo do ICM

O ICM é calculado por imagem como:

```
ICM(imagem) = Σ  peso(classᵢ) × conf(detecção_i)
              i ∈ detecções
```

A escala de interpretação adotada:

| ICM          | Condição          | Ação recomendada                  |
|--------------|-------------------|-----------------------------------|
| 0 – 2        | Boa               | Manutenção preventiva             |
| 2 – 5        | Regular           | Monitoramento frequente           |
| 5 – 10       | Ruim              | Intervenção em curto prazo        |
| > 10         | Crítica           | Intervenção urgente               |

---

## 4. Resultados

### 4.1 Métricas Globais

| Métrica       | Valor      |
|---------------|------------|
| mAP@50        | **72,54%** |
| mAP@50-95     | **63,22%** |
| Precision     | **76,14%** |
| Recall        | **77,39%** |

### 4.2 Análise dos Resultados

- O modelo atingiu **recall de 77,4%**, indicando boa sensibilidade na detecção de patologias — importante para não deixar defeitos críticos sem detecção.
- A **precision de 76,1%** demonstra baixa taxa de falsos positivos.
- O mAP@50-95 de **63,2%** reflete robustez na localização espacial das detecções em diferentes limiares de IoU.
- Classes com maior variação visual (`trinca`, `remendo`) tendem a apresentar métricas individualmente inferiores à média global.

### 4.3 Arquivos de Saída

| Arquivo                              | Descrição                            |
|--------------------------------------|--------------------------------------|
| `models/pavimentos_yolov8_best.pt`   | Melhor checkpoint PyTorch            |
| `models/pavimentos_yolov8.onnx`      | Modelo exportado para inferência     |
| `models/pavimentos_yolov8.torchscript` | Modelo para deploy em produção    |
| `results/predicoes_teste.csv`        | Predições detalhadas no conjunto de teste |
| `results/figures/`                   | Gráficos de métricas, matriz de confusão, curvas PR |

---

## 5. Exportação e Deploy

O script `05_export.py` gera três formatos a partir do melhor checkpoint:

```python
# ONNX — inferência cross-platform
model.export(format="onnx", imgsz=640, simplify=True)

# TorchScript — deploy C++/mobile
model.export(format="torchscript", imgsz=640)
```

### Inferência com ONNX (exemplo)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/pavimentos_yolov8.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_tensor})
```

---

## 6. Limitações e Trabalhos Futuros

- **Variabilidade climática:** o dataset atual tem cobertura limitada de imagens sob chuva ou à noite.
- **Resolução das câmeras:** modelos capturados com smartphones de baixa resolução podem reduzir precision em `trinca`.
- **Trabalhos futuros:**
  - Fine-tuning com dados de municípios específicos
  - Integração com sistema GIS para mapeamento georreferenciado de ICM
  - Segmentação semântica para estimativa de área afetada
  - Versão embarcada (YOLOv8n) para dispositivos de borda em viaturas de inspeção

---

## 7. Referências

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- DNIT (2003). *Manual de Restauração de Pavimentos Asfálticos*. Departamento Nacional de Infraestrutura de Transportes.
- Redmon, J. & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*. arXiv:1804.02767.
