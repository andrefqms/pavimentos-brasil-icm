import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

import config

warnings.filterwarnings('ignore')


def carregar_modelo():
    path_file = config.MODEL_DIR / 'best_weights_path.txt'
    if path_file.exists():
        best_weights = Path(path_file.read_text().strip())
    else:
        candidates = list((config.MODEL_DIR / 'runs').rglob('best.pt'))
        if not candidates:
            print('ERRO: Nenhum modelo treinado encontrado.')
            print('Execute primeiro: python 03_treinar.py')
            sys.exit(1)
        best_weights = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    print(f'Modelo: {best_weights}')
    return YOLO(str(best_weights)), best_weights


def avaliar_teste(model, yaml_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('\nAvaliando no conjunto de teste...')
    metricas = model.val(
        data    = str(yaml_path),
        split   = 'test',
        imgsz   = config.IMG_SIZE,
        conf    = config.CONF_THRESH,
        iou     = config.IOU_THRESH,
        device  = device,
        verbose = True,
        workers = 0,
    )

    print(f'\nResultados:')
    print(f'  mAP@0.50      : {metricas.box.map50:.4f}')
    print(f'  mAP@0.50:0.95  : {metricas.box.map:.4f}')
    print(f'  Precision      : {metricas.box.mp:.4f}')
    print(f'  Recall         : {metricas.box.mr:.4f}')

    return metricas


def plotar_ap_por_classe(metricas):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_class_ap50 = metricas.box.ap50
    classes_com_ap = config.CLASS_NAMES[:len(per_class_ap50)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(classes_com_ap, per_class_ap50,
                   color=sns.color_palette('husl', len(per_class_ap50)))
    ax.axvline(x=metricas.box.map50, color='red', linestyle='--', linewidth=1.5,
               label=f'mAP@50 medio = {metricas.box.map50:.3f}')
    ax.set_xlabel('AP@0.50')
    ax.set_title('Average Precision por Classe (AP@0.50)',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, per_class_ap50):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    out = config.RESULTS_DIR / 'yolo_ap_por_classe.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'AP por classe salvo: {out}')


def inferencia_visual(model):
    test_imgs = list((config.YOLO_DATA_DIR / 'images' / 'test').glob('*.*'))
    amostras  = test_imgs[:6]

    if not amostras:
        print('Sem imagens de teste para inferencia visual.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import cv2
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, img_path in zip(axes.flatten(), amostras):
        result    = model.predict(
            source  = str(img_path),
            conf    = config.CONF_THRESH,
            iou     = config.IOU_THRESH,
            device  = device,
            verbose = False,
        )[0]

        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        n_det = len(result.boxes)
        ax.imshow(annotated)
        ax.set_title(f'{img_path.name}\n{n_det} deteccao(oes)', fontsize=9)
        ax.axis('off')

    plt.suptitle('Inferencia YOLOv8 — Conjunto de Teste',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = config.RESULTS_DIR / 'yolo_inferencia_teste.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Inferencia visual salva: {out}')


def calcular_icm(img_path, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    result   = model.predict(
        source  = str(img_path),
        conf    = config.CONF_THRESH,
        iou     = config.IOU_THRESH,
        device  = device,
        verbose = False,
    )[0]

    img      = Image.open(img_path)
    img_area = img.width * img.height

    defeitos = {}
    for box in result.boxes:
        cls_idx  = int(box.cls.item())
        cls_name = config.CLASS_NAMES[cls_idx]
        xyxy     = box.xyxy[0].cpu().numpy()
        area_px  = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        area_pct = area_px / img_area * 100

        if cls_name not in defeitos:
            defeitos[cls_name] = {'count': 0, 'area_total_pct': 0.0}
        defeitos[cls_name]['count']          += 1
        defeitos[cls_name]['area_total_pct'] += area_pct

    penalidade = sum(
        config.ICM_PESO.get(cls, 1) * info['area_total_pct']
        for cls, info in defeitos.items()
    )

    icm_score = max(0, 100 - penalidade)

    if icm_score >= 80:   condicao = 'Otimo'
    elif icm_score >= 60: condicao = 'Bom'
    elif icm_score >= 40: condicao = 'Regular'
    elif icm_score >= 20: condicao = 'Ruim'
    else:                 condicao = 'Pessimo'

    return {
        'imagem'      : Path(img_path).name,
        'icm_score'   : round(icm_score, 2),
        'condicao'    : condicao,
        'defeitos'    : defeitos,
        'n_deteccoes' : len(result.boxes),
    }


def calcular_icm_conjunto(model):
    test_imgs = list((config.YOLO_DATA_DIR / 'images' / 'test').glob('*.*'))

    if not test_imgs:
        print('Sem imagens de teste para ICM.')
        return pd.DataFrame()

    icm_results = []
    for img_path in tqdm(test_imgs, desc='Calculando ICM'):
        try:
            icm_results.append(calcular_icm(img_path, model))
        except Exception as e:
            print(f'Erro ICM {img_path.name}: {e}')

    if not icm_results:
        print('Nenhum resultado de ICM gerado.')
        return pd.DataFrame()

    icm_df = pd.DataFrame([{
        'imagem'      : r['imagem'],
        'icm_score'   : r['icm_score'],
        'condicao'    : r['condicao'],
        'n_deteccoes' : r['n_deteccoes'],
    } for r in icm_results])

    print(icm_df.to_string(index=False))
    print(f'\nICM medio do trecho: {icm_df["icm_score"].mean():.2f}')

    condicao_cores = {
        'Otimo'   : '#2ecc71',
        'Bom'     : '#27ae60',
        'Regular' : '#f39c12',
        'Ruim'    : '#e67e22',
        'Pessimo' : '#e74c3c',
    }

    fig, ax = plt.subplots(figsize=(max(12, len(icm_df) * 1.5), 5))
    cores = [condicao_cores.get(c, 'gray') for c in icm_df['condicao']]
    bars  = ax.bar(icm_df['imagem'], icm_df['icm_score'],
                   color=cores, edgecolor='white')

    ax.axhline(y=80, color='green',  linestyle='--', linewidth=1,
               alpha=0.7, label='Otimo (80)')
    ax.axhline(y=60, color='orange', linestyle='--', linewidth=1,
               alpha=0.7, label='Regular (60)')
    ax.axhline(y=40, color='red',    linestyle='--', linewidth=1,
               alpha=0.7, label='Ruim (40)')

    ax.set_xlabel('Imagem')
    ax.set_ylabel('ICM Score')
    ax.set_title('Indice de Condicao de Manutencao (ICM) por Imagem',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='lower left')

    for bar, row in zip(bars, icm_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{row.icm_score:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    out = config.RESULTS_DIR / 'icm_scores.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Grafico ICM salvo: {out}')

    icm_df.to_csv(config.RESULTS_DIR / 'resultados_icm.csv', index=False)
    return icm_df


def exportar(model, best_weights, metricas, icm_df):
    onnx_path = config.MODEL_DIR / 'pavimentos_yolov8.onnx'

    print('\nExportando para ONNX...')
    model.export(
        format   = 'onnx',
        imgsz    = config.IMG_SIZE,
        opset    = 12,
        simplify = True,
        dynamic  = False,
    )

    onnx_src = Path(str(best_weights).replace('.pt', '.onnx'))
    if onnx_src.exists():
        import shutil
        shutil.copy(onnx_src, onnx_path)

    metadata = {
        'modelo'      : config.YOLO_MODEL,
        'arquitetura' : 'YOLOv8',
        'dataset'     : 'Pavimentos-Brasil (Kaggle)',
        'classes'     : config.CLASS_NAMES,
        'num_classes' : config.NUM_CLASSES,
        'img_size'    : config.IMG_SIZE,
        'conf_thresh' : config.CONF_THRESH,
        'iou_thresh'  : config.IOU_THRESH,
        'mAP50'       : float(metricas.box.map50),
        'mAP50_95'    : float(metricas.box.map),
        'precision'   : float(metricas.box.mp),
        'recall'      : float(metricas.box.mr),
        'pesos_icm'   : config.ICM_PESO,
    }

    meta_path = config.MODEL_DIR / 'model_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f'\nArquivos exportados:')
    print(f'  Modelo ONNX   : {onnx_path}')
    print(f'  Metadados     : {meta_path}')
    print(f'  CSV ICM       : {config.RESULTS_DIR / "resultados_icm.csv"}')
    print(f'  Figuras       : {config.RESULTS_DIR}')


def main():
    print('--- Avaliacao e ICM ---\n')

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    yaml_path = config.YOLO_DATA_DIR / 'data.yaml'
    if not yaml_path.exists():
        print('ERRO: data.yaml nao encontrado.')
        print('Execute primeiro: python 02_preparar_dataset.py')
        sys.exit(1)

    model, best_weights = carregar_modelo()

    metricas = avaliar_teste(model, yaml_path)
    plotar_ap_por_classe(metricas)
    inferencia_visual(model)
    icm_df = calcular_icm_conjunto(model)
    exportar(model, best_weights, metricas, icm_df)

    print('\nPipeline concluido.')
    print(f'Resultados em: {config.RESULTS_DIR}')


if __name__ == '__main__':
    main()
