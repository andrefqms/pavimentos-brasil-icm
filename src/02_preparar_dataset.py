import random
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

import config


def criar_estrutura():
    for split in ['train', 'val', 'test']:
        (config.YOLO_DATA_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (config.YOLO_DATA_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f'Estrutura criada em: {config.YOLO_DATA_DIR}')


def fazer_split():
    label_files = list((config.YOLO_DATA_DIR / 'labels' / 'train').glob('*.txt'))

    if not label_files:
        print('ERRO: Nenhum label encontrado em labels/train.')
        print('Execute primeiro: python 01_pseudo_labels.py')
        sys.exit(1)

    print(f'Total de arquivos anotados: {len(label_files):,}')

    train_files, temp_files = train_test_split(
        label_files, test_size=0.30, random_state=config.SEED
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=config.SEED
    )

    def mover_split(label_files_list, split_name):
        if split_name == 'train':
            return
        lbl_dst = config.YOLO_DATA_DIR / 'labels'  / split_name
        img_dst = config.YOLO_DATA_DIR / 'images'  / split_name
        lbl_dst.mkdir(parents=True, exist_ok=True)
        img_dst.mkdir(parents=True, exist_ok=True)

        for lbl in label_files_list:
            for ext in ['.jpg', '.jpeg', '.png']:
                img_src = config.YOLO_DATA_DIR / 'images' / 'train' / (lbl.stem + ext)
                if img_src.exists():
                    shutil.move(str(img_src), img_dst / img_src.name)
                    break
            shutil.move(str(lbl), lbl_dst / lbl.name)

    mover_split(val_files,  'val')
    mover_split(test_files, 'test')

    n_train = len(list((config.YOLO_DATA_DIR / 'labels' / 'train').glob('*.txt')))
    n_val   = len(val_files)
    n_test  = len(test_files)

    print(f'\nSplit realizado:')
    print(f'  Treino    : {n_train:,}')
    print(f'  Validacao : {n_val:,}')
    print(f'  Teste     : {n_test:,}')

    return n_train, n_val, n_test


def gerar_yaml():
    yaml_path = config.YOLO_DATA_DIR / 'data.yaml'
    conteudo  = {
        'path'  : str(config.YOLO_DATA_DIR),
        'train' : 'images/train',
        'val'   : 'images/val',
        'test'  : 'images/test',
        'nc'    : config.NUM_CLASSES,
        'names' : config.CLASS_NAMES,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(conteudo, f, default_flow_style=False, allow_unicode=True)
    print(f'\ndata.yaml gerado: {yaml_path}')
    return yaml_path


def preview_anotacoes():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_imgs = list((config.YOLO_DATA_DIR / 'images' / 'train').glob('*.*'))
    amostras   = random.sample(train_imgs, min(4, len(train_imgs)))

    if not amostras:
        print('Sem imagens para preview.')
        return

    colors = plt.cm.get_cmap('tab10', config.NUM_CLASSES).colors
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, img_path in zip(axes.flatten(), amostras):
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        ax.imshow(img)

        lbl_path = config.YOLO_DATA_DIR / 'labels' / 'train' / (img_path.stem + '.txt')
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1   = (cx - bw / 2) * w
                y1   = (cy - bh / 2) * h
                rect = patches.Rectangle(
                    (x1, y1), bw * w, bh * h,
                    linewidth=2,
                    edgecolor=colors[cls_idx % len(colors)],
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 4, config.CLASS_NAMES[cls_idx],
                        color='white', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor=colors[cls_idx % len(colors)], alpha=0.8, pad=1))
        ax.axis('off')
        ax.set_title(img_path.name, fontsize=9)

    plt.suptitle('Preview das Anotacoes YOLO', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = config.RESULTS_DIR / 'anotacoes_preview.png'
    plt.savefig(out, bbox_inches='tight', dpi=120)
    plt.close()
    print(f'Preview salvo: {out}')


def main():
    print('--- Preparando dataset YOLO ---\n')
    criar_estrutura()
    fazer_split()
    yaml_path = gerar_yaml()
    preview_anotacoes()
    print(f'\nProximo passo: python 03_treinar.py')


if __name__ == '__main__':
    main()
