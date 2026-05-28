import os
import sys
import shutil
import tempfile
import warnings
from glob import glob
from pathlib import Path

import cv2
import torch
import torchvision.ops as ops
from tqdm import tqdm

import config

warnings.filterwarnings('ignore')


def baixar_gdino_ckpt():
    import urllib.request
    url  = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
    dest = config.GDINO_CKPT
    if not dest.exists():
        print(f'Baixando pesos do Grounding DINO (~700 MB)...')
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest, reporthook=lambda b, bs, t: print(
            f'\r{min(100, int(b*bs/t*100))}%', end='', flush=True) if t > 0 else None)
        print('\nDownload concluido.')
    else:
        print(f'Pesos ja existem: {dest}')


def baixar_gdino_config():
    import urllib.request
    url  = 'https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    dest = config.GDINO_CONFIG
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print('Baixando config do Grounding DINO...')
        urllib.request.urlretrieve(url, dest)
        print('Config baixado.')
    else:
        print(f'Config ja existe: {dest}')


def apply_nms(boxes, scores):
    if len(boxes) == 0:
        return []
    keep = ops.nms(boxes, scores, config.NMS_THRESHOLD)
    return keep.tolist()


def generate_pseudo_labels(img_path, gdino_model, load_image_fn, predict_fn, tmp_dir):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f'Erro lendo: {img_path}')
        return []

    image = cv2.resize(image, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))

    tmp_path = Path(tmp_dir) / 'temp_img.jpg'
    cv2.imwrite(str(tmp_path), image)

    image_source, image_tensor = load_image_fn(str(tmp_path))

    all_lines = []

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        prompt = config.TEXT_PROMPTS[class_name]
        try:
            boxes, logits, _ = predict_fn(
                model          = gdino_model,
                image          = image_tensor,
                caption        = prompt,
                box_threshold  = config.BOX_THRESHOLD,
                text_threshold = config.TEXT_THRESHOLD,
            )

            if len(boxes) == 0:
                continue

            xyxy_boxes, valid_scores = [], []

            for box, score in zip(boxes, logits):
                cx, cy, bw, bh = box.tolist()
                if bw * bh < config.MIN_BOX_AREA:
                    continue
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                xyxy_boxes.append([x1, y1, x2, y2])
                valid_scores.append(float(score))

            if not xyxy_boxes:
                continue

            xyxy_t  = torch.tensor(xyxy_boxes,   dtype=torch.float32)
            scores_t = torch.tensor(valid_scores, dtype=torch.float32)
            keep    = apply_nms(xyxy_t, scores_t)

            for idx in keep:
                cx, cy, bw, bh = boxes[idx].tolist()
                all_lines.append(f'{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

        except Exception as e:
            print(f'Erro: {img_path.name} | {class_name} | {e}')

    return all_lines


def main():
    baixar_gdino_ckpt()
    baixar_gdino_config()

    try:
        from groundingdino.util.inference import load_model, load_image, predict
    except ImportError:
        print('ERRO: groundingdino nao instalado.')
        print('Execute: pip install groundingdino-py transformers==4.41.0')
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Dispositivo: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print('Carregando Grounding DINO...')
    gdino_model = load_model(str(config.GDINO_CONFIG), str(config.GDINO_CKPT))
    print('Modelo carregado.')

    pseudo_img_dir = config.YOLO_DATA_DIR / 'images' / 'train'
    pseudo_lbl_dir = config.YOLO_DATA_DIR / 'labels' / 'train'
    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_images = (
        glob(str(config.DATA_DIR / '**' / '*.jpg'),  recursive=True) +
        glob(str(config.DATA_DIR / '**' / '*.jpeg'), recursive=True) +
        glob(str(config.DATA_DIR / '**' / '*.png'),  recursive=True)
    )

    print(f'Total de imagens: {len(all_images):,}')

    if len(all_images) == 0:
        print(f'ERRO: Nenhuma imagem encontrada em {config.DATA_DIR}')
        sys.exit(1)

    total_gerados = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for img_path in tqdm(all_images, desc='Gerando pseudo-labels'):
            try:
                img_path = Path(img_path)
                lines    = generate_pseudo_labels(img_path, gdino_model, load_image, predict, tmp_dir)

                if not lines:
                    continue

                dest_img = pseudo_img_dir / img_path.name
                dest_lbl = pseudo_lbl_dir / (img_path.stem + '.txt')

                shutil.copy(img_path, dest_img)
                dest_lbl.write_text('\n'.join(lines))
                total_gerados += 1

            except Exception as e:
                print(f'Erro geral: {img_path} | {e}')

    print(f'\nPseudo-labels gerados: {total_gerados}')

    labels_gerados = list(pseudo_lbl_dir.glob('*.txt'))
    print(f'Labels salvos: {len(labels_gerados)}')

    if not labels_gerados:
        print('ATENCAO: Nenhum pseudo-label foi gerado.')
    else:
        print('Pseudo-labeling concluido.')
        print(f'Imagens em: {pseudo_img_dir}')
        print(f'Labels em : {pseudo_lbl_dir}')
        print('\nProximo passo: python 02_preparar_dataset.py')


if __name__ == '__main__':
    main()
