# Pavimentos Brasil — Detecção de danos em rodovias com YOLOv8 + ICM

[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/model-YOLOv8m-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Detecção automática de patologias em pavimentos urbanos brasileiros usando YOLOv8m, com cálculo do **Índice de Condição da Manutenção (ICM)** por imagem.

---

## Visão Geral

Este projeto treina e avalia um modelo YOLOv8 para identificar 7 classes de defeitos e elementos de infraestrutura em vias públicas brasileiras. O ICM é calculado a partir das detecções, ponderando cada classe por seu grau de criticidade.

### Classes Detectadas

| Classe                  | Peso ICM | Descrição                          |
|-------------------------|----------|------------------------------------|
| `buraco`                | 5        | Afundamento crítico do pavimento   |
| `trinca`                | 3        | Fissuras superficiais              |
| `remendo`               | 2        | Reparo temporário existente        |
| `drenagem`              | 2        | Problemas de escoamento            |
| `sinalizacao_vertical`  | 1        | Placas e sinais verticais          |
| `sinalizacao_horizontal`| 1        | Faixas e marcações no piso         |
| `vegetacao`             | 1        | Vegetação invasora                 |

### Métricas do Modelo Treinado

| Métrica       | Valor  |
|---------------|--------|
| mAP@50        | 72.54% |
| mAP@50-95     | 63.22% |
| Precision     | 76.14% |
| Recall        | 77.39% |

---

## Estrutura do Projeto seguindo as boas praticas na literatura

```
pavimentos-brasil-icm/
├── data/
│   ├── raw/          <- imagens originais (NUNCA modificar)
│   └── processed/    <- splits gerados pelo script
├── src/
│   ├── 01_eda.py              <- análise exploratória
│   ├── 02_preprocessing.py    <- geração de splits
│   ├── 03_train.py            <- treinamento YOLOv8
│   ├── 04_evaluate.py         <- avaliação e ICM
│   └── 05_export.py           <- exportação de modelos
├── results/
│   ├── figures/               <- gráficos salvos
│   └── predicoes_teste.csv    <- predições no conjunto de teste
├── models/                    <- checkpoints .pt
├── docs/
│   └── estudo_de_caso_pavimentos_yolov8.md
├── config.py
├── requirements.txt
├── setup.bat
└── README.md
```

---

## Início Rápido

### 1. Pré-requisitos

- Python 3.12
- CUDA 11.8+ (recomendado para treino com GPU, neste caso foi usado uma RTX 3060)

### 2. Instalação (Windows)

```bat
setup.bat
```

Ou manualmente:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
# edite .env com seus caminhos e chaves
```

### 4. Executar pipeline completo

```bash
python src/01_eda.py
python src/02_preprocessing.py
python src/03_train.py
python src/04_evaluate.py
python src/05_export.py
```

---

## Modelos Exportados

Os modelos treinados estão disponíveis em três formatos:

| Formato        | Arquivo                              | Uso recomendado              |
|----------------|--------------------------------------|------------------------------|
| PyTorch        | `pavimentos_yolov8_best.pt`          | Retreino / fine-tuning       |
| ONNX           | `pavimentos_yolov8.onnx`             | Inferência cross-platform    |
| TorchScript    | `pavimentos_yolov8_torchscript.ts`   | Deploy em produção C++/mobile|



## Licença

MIT © 2026 — veja [LICENSE](LICENSE) para detalhes.
