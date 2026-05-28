# Pavimentos Brasil — Detecção de Danos em Rodovias com YOLOv8 + ICM

[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/model-YOLOv8m-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema de detecção automática de patologias em pavimentos urbanos brasileiros utilizando YOLOv8m e pseudo-labeling com GroundingDINO, com cálculo estimado do **Índice de Condição da Manutenção (ICM)** por imagem.

---

## Sobre o Dataset

O projeto utiliza o dataset **PavimentosBrasil**, composto por aproximadamente 9.000 imagens capturadas por smartphone em rodovias dos estados do Ceará e Piauí, Brasil.

O conjunto de dados foi desenvolvido para apoiar pesquisas em visão computacional aplicada à infraestrutura rodoviária, permitindo a identificação automática de defeitos como:

* buracos
* trincas
* remendos
* problemas de drenagem
* sinalização vertical e horizontal
* vegetação lateral

Dataset disponível em:

* [PavimentosBrasil Dataset](https://www.kaggle.com/datasets/mateusserafim/pavimentosbrasil?utm_source=chatgpt.com)

---

## Visão Geral

O pipeline realiza:

1. geração automática de pseudo-labels utilizando GroundingDINO
2. treinamento de um detector YOLOv8m
3. inferência sobre imagens de rodovias
4. cálculo aproximado do ICM baseado nas áreas detectadas

O objetivo é investigar a viabilidade do uso de inteligência artificial para inspeção automatizada de pavimentos rodoviários brasileiros.

---

## Classes Detectadas

| Classe                   | Peso ICM | Descrição                        |
| ------------------------ | -------- | -------------------------------- |
| `buraco`                 | 5        | Afundamento crítico do pavimento |
| `trinca`                 | 3        | Fissuras superficiais            |
| `remendo`                | 2        | Reparo temporário existente      |
| `drenagem`               | 2        | Problemas de escoamento          |
| `sinalizacao_vertical`   | 1        | Placas e sinais verticais        |
| `sinalizacao_horizontal` | 1        | Faixas e marcações no piso       |
| `vegetacao`              | 1        | Vegetação invasora               |

---

## Métricas do Modelo

| Métrica   | Valor  |
| --------- | ------ |
| mAP@50    | 72.54% |
| mAP@50-95 | 63.22% |
| Precision | 76.14% |
| Recall    | 77.39% |

Treinamento realizado com:

* YOLOv8m
* 50 épocas máximas
* early stopping (`patience=15`)
* GPU RTX 3060

---

## Limitações Conhecidas

Este projeto utiliza pseudo-labels gerados automaticamente pelo GroundingDINO, o que pode introduzir falsas detecções e anotações imprecisas. O modelo também apresenta limitações na detecção de defeitos pequenos, como trincas finas e desgastes superficiais, já que o problema é que ele foi treinado em imagens gerais da internet, não em rodovias brasileiras em especifico. Além disso, o cálculo do ICM implementado neste trabalho é uma estimativa simplificada baseada na área detectada dos danos e não substitui avaliações técnicas oficiais realizadas por especialistas.

---

## Estrutura do Projeto

```text
pavimentos-brasil-icm/
├── data/
│   └── Kaggle/
│   └── raw data and processed -> https://zenodo.org/records/20433455
├── src/
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   └── 05_export.py
├── results/
│   ├── figures/
│   └── predicoes_teste.csv
├── models/
├── docs/
├── config.py
├── requirements.txt
├── setup.bat
└── README.md
```

---

## Instalação

### Pré-requisitos

* Python 3.12
* CUDA 11.8+ (recomendado)
* GPU NVIDIA (RTX 3060 utilizada nos experimentos)

### Instalação automática

```bash
setup.bat
```

### Instalação manual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Execução do Pipeline

```bash
python src/01_eda.py
python src/02_preprocessing.py
python src/03_train.py
python src/04_evaluate.py
python src/05_export.py
```

---

## Modelos Exportados

| Formato     | Arquivo                            | Uso                       |
| ----------- | ---------------------------------- | ------------------------- |
| PyTorch     | `pavimentos_yolov8_best.pt`        | Fine-tuning               |
| ONNX        | `pavimentos_yolov8.onnx`           | Inferência cross-platform |
| TorchScript | `pavimentos_yolov8_torchscript.ts` | Deploy em produção        |

---

## Tecnologias Utilizadas

* Python
* YOLOv8
* GroundingDINO
* PyTorch
* OpenCV
* Ultralytics
* NumPy
* Matplotlib

---

## Licença

MIT © 2026 — veja [LICENSE](LICENSE) para mais detalhes.
