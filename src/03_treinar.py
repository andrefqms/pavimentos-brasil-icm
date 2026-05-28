import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

import config

warnings.filterwarnings('ignore')


def verificar_dataset():
    yaml_path = config.YOLO_DATA_DIR / 'data.yaml'
    if not yaml_path.exists():
        print('ERRO: data.yaml nao encontrado.')
        print('Execute primeiro: python 02_preparar_dataset.py')
        sys.exit(1)

    n_train = len(list((config.YOLO_DATA_DIR / 'labels' / 'train').glob('*.txt')))
    n_val   = len(list((config.YOLO_DATA_DIR / 'labels' / 'val').glob('*.txt')))

    if n_train == 0:
        print('ERRO: Nenhum label de treino encontrado.')
        sys.exit(1)

    print(f'Treino : {n_train:,} imagens anotadas')
    print(f'Val    : {n_val:,} imagens anotadas')
    return yaml_path


def treinar(yaml_path):
    from ultralytics import YOLO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nDispositivo: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'VRAM: {vram:.1f} GB')
    else:
        print('AVISO: GPU nao detectada. Treinamento sera muito lento.')

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.YOLO_MODEL)
    print(f'\nModelo base: {config.YOLO_MODEL}')

    print(f'\nIniciando treinamento...')
    print(f'  Epocas   : {config.EPOCHS}')
    print(f'  Batch    : {config.BATCH_SIZE}')
    print(f'  Imagem   : {config.IMG_SIZE}px')

    results = model.train(
        data         = str(yaml_path),
        epochs       = config.EPOCHS,
        imgsz        = config.IMG_SIZE,
        batch        = config.BATCH_SIZE,
        device       = device,
        project      = str(config.MODEL_DIR / 'runs'),
        name         = 'pavimentos_yolov8',
        seed         = config.SEED,
        patience     = 15,
        lr0          = 0.01,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs   = 3,
        warmup_momentum = 0.8,
        box          = 7.5,
        cls          = 0.5,
        dfl          = 1.5,
        flipud       = 0.0,
        fliplr       = 0.5,
        mosaic       = 1.0,
        mixup        = 0.1,
        copy_paste   = 0.1,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        degrees      = 10.0,
        translate    = 0.1,
        scale        = 0.5,
        shear        = 2.0,
        save         = True,
        save_period  = 10,
        plots        = True,
        verbose      = True,
        workers      = 0,
    )

    return results


def plotar_curvas(results):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_csv = Path(results.save_dir) / 'results.csv'
    if not results_csv.exists():
        print('results.csv nao encontrado, pulando curvas.')
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    loss_metrics = [
        ('train/box_loss', 'val/box_loss', 'Box Loss'),
        ('train/cls_loss', 'val/cls_loss', 'Class Loss'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Curvas de Perda de Treino e Validacao — YOLOv8',
                 fontsize=16, fontweight='bold')

    for ax, (train_col, val_col, title) in zip(axes, loss_metrics):
        if train_col in df.columns and val_col in df.columns:
            ax.plot(df['epoch'], df[train_col], label=f'Treino {title}',
                    color='steelblue', linewidth=2)
            ax.plot(df['epoch'], df[val_col],   label=f'Validacao {title}',
                    color='coral',     linewidth=2, linestyle='--')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoca')
            ax.set_ylabel('Perda')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Dados nao encontrados: {title}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = config.RESULTS_DIR / 'yolo_loss_curves.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Curvas salvas: {out}')

    perf_metrics = [
        ('metrics/mAP50(B)',    'metrics/mAP50-95(B)', 'mAP'),
        ('metrics/precision(B)', 'metrics/recall(B)',   'Precision / Recall'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Curvas de Treinamento — YOLOv8', fontsize=14, fontweight='bold')

    for ax, (col1, col2, title) in zip(axes, perf_metrics):
        if col1 in df.columns:
            ax.plot(df['epoch'], df[col1], label=col1.split('/')[-1],
                    color='steelblue', linewidth=2)
        if col2 in df.columns:
            ax.plot(df['epoch'], df[col2], label=col2.split('/')[-1],
                    color='coral', linewidth=2, linestyle='--')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoca')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out2 = config.RESULTS_DIR / 'yolo_training_curves.png'
    plt.savefig(out2, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Curvas de desempenho salvas: {out2}')


def main():
    print('--- Treinamento YOLOv8 ---\n')

    yaml_path = verificar_dataset()
    results   = treinar(yaml_path)

    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f'\nMelhor modelo: {best_weights}')

    with open(config.MODEL_DIR / 'best_weights_path.txt', 'w') as f:
        f.write(str(best_weights))

    plotar_curvas(results)

    print(f'\nProximo passo: python 04_avaliar.py')


if __name__ == '__main__':
    main()
