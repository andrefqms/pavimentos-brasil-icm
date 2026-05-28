"""
05_export.py — Exportacao do Modelo Treinado
=============================================
Exporta o melhor modelo YOLOv8 para multiplos formatos de deployment,
gera um relatorio HTML completo e empacota os artefatos finais.

Formatos exportados:
  - ONNX      (deployment multiplataforma)
  - TorchScript / TorchScript quantizado (CPU otimizado)
  - Copia do .pt original (best.pt)

Artefatos gerados em MODEL_DIR/export/:
  - pavimentos_yolov8.onnx
  - pavimentos_yolov8_torchscript.torchscript
  - pavimentos_yolov8_best.pt
  - model_metadata.json
  - relatorio_final.html
  - pacote_deploy.zip

Pre-requisito: python 04_avaliar.py ja ter sido executado.
"""

import json
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import config

warnings.filterwarnings('ignore')

EXPORT_DIR = config.MODEL_DIR / 'export'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def carregar_modelo() -> tuple[YOLO, Path]:
    """Localiza e carrega o melhor modelo treinado."""
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

    if not best_weights.exists():
        print(f'ERRO: Arquivo de pesos nao encontrado: {best_weights}')
        sys.exit(1)

    print(f'Modelo carregado: {best_weights}')
    return YOLO(str(best_weights)), best_weights


def carregar_metricas() -> dict:
    """Le metricas salvas pelo 04_avaliar.py (model_metadata.json)."""
    meta_path = config.MODEL_DIR / 'model_metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    print('AVISO: model_metadata.json nao encontrado. Metricas nao incluidas no relatorio.')
    return {}


def carregar_icm_csv() -> pd.DataFrame:
    """Le resultados de ICM gerados pelo 04_avaliar.py."""
    csv_path = config.RESULTS_DIR / 'resultados_icm.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    print('AVISO: resultados_icm.csv nao encontrado.')
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Exportacao de formatos
# ---------------------------------------------------------------------------

def exportar_onnx(model: YOLO, best_weights: Path) -> Path | None:
    """Exporta o modelo para ONNX (opset 12, simplificado)."""
    print('\n[1/3] Exportando para ONNX...')
    try:
        model.export(
            format='onnx',
            imgsz=config.IMG_SIZE,
            opset=12,
            simplify=True,
            dynamic=False,
        )
        src = Path(str(best_weights).replace('.pt', '.onnx'))
        dst = EXPORT_DIR / 'pavimentos_yolov8.onnx'
        if src.exists():
            shutil.copy(src, dst)
            tamanho_mb = dst.stat().st_size / 1024 ** 2
            print(f'    ONNX salvo: {dst}  ({tamanho_mb:.1f} MB)')
            return dst
        else:
            print(f'    AVISO: arquivo ONNX nao encontrado em {src}')
            return None
    except Exception as e:
        print(f'    ERRO na exportacao ONNX: {e}')
        return None


def exportar_torchscript(model: YOLO, best_weights: Path) -> Path | None:
    """Exporta o modelo para TorchScript."""
    print('\n[2/3] Exportando para TorchScript...')
    try:
        model.export(
            format='torchscript',
            imgsz=config.IMG_SIZE,
        )
        src = Path(str(best_weights).replace('.pt', '.torchscript'))
        dst = EXPORT_DIR / 'pavimentos_yolov8_torchscript.torchscript'
        if src.exists():
            shutil.copy(src, dst)
            tamanho_mb = dst.stat().st_size / 1024 ** 2
            print(f'    TorchScript salvo: {dst}  ({tamanho_mb:.1f} MB)')
            return dst
        else:
            print(f'    AVISO: arquivo TorchScript nao encontrado em {src}')
            return None
    except Exception as e:
        print(f'    ERRO na exportacao TorchScript: {e}')
        return None


def copiar_pesos(best_weights: Path) -> Path:
    """Copia o best.pt original para o diretorio de export."""
    print('\n[3/3] Copiando pesos originais (.pt)...')
    dst = EXPORT_DIR / 'pavimentos_yolov8_best.pt'
    shutil.copy(best_weights, dst)
    tamanho_mb = dst.stat().st_size / 1024 ** 2
    print(f'    Pesos copiados: {dst}  ({tamanho_mb:.1f} MB)')
    return dst


# ---------------------------------------------------------------------------
# Metadata JSON
# ---------------------------------------------------------------------------

def salvar_metadata(metricas: dict, arquivos_exportados: dict) -> Path:
    """Gera/atualiza o model_metadata.json com info de exportacao."""
    metadata = {
        'modelo'         : config.YOLO_MODEL,
        'arquitetura'    : 'YOLOv8',
        'dataset'        : 'Pavimentos-Brasil',
        'classes'        : config.CLASS_NAMES,
        'num_classes'    : config.NUM_CLASSES,
        'img_size'       : config.IMG_SIZE,
        'conf_thresh'    : config.CONF_THRESH,
        'iou_thresh'     : config.IOU_THRESH,
        'pesos_icm'      : config.ICM_PESO,
        'exportado_em'   : datetime.now().isoformat(timespec='seconds'),
        'arquivos'       : {k: str(v) for k, v in arquivos_exportados.items() if v},
    }

    # mescla metricas se disponiveis
    for campo in ('mAP50', 'mAP50_95', 'precision', 'recall'):
        if campo in metricas:
            metadata[campo] = metricas[campo]

    dst = EXPORT_DIR / 'model_metadata.json'
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f'\nMetadados salvos: {dst}')
    return dst


# ---------------------------------------------------------------------------
# Relatorio HTML
# ---------------------------------------------------------------------------

def _grafico_icm_base64(icm_df: pd.DataFrame) -> str:
    """Gera grafico ICM e retorna como string base64 para embed no HTML."""
    import base64, io

    condicao_cores = {
        'Otimo': '#2ecc71', 'Bom': '#27ae60',
        'Regular': '#f39c12', 'Ruim': '#e67e22', 'Pessimo': '#e74c3c',
    }
    cores = [condicao_cores.get(c, 'gray') for c in icm_df['condicao']]

    fig, ax = plt.subplots(figsize=(max(10, len(icm_df) * 1.2), 4))
    bars = ax.bar(icm_df['imagem'], icm_df['icm_score'], color=cores, edgecolor='white')
    ax.axhline(80, color='green',  linestyle='--', linewidth=1, alpha=0.6, label='Otimo (80)')
    ax.axhline(60, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Regular (60)')
    ax.axhline(40, color='red',    linestyle='--', linewidth=1, alpha=0.6, label='Ruim (40)')
    ax.set_ylim(0, 115)
    ax.set_ylabel('ICM Score')
    ax.set_title('ICM por Imagem')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)
    for bar, row in zip(bars, icm_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{row.icm_score:.0f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def gerar_relatorio_html(metricas: dict, icm_df: pd.DataFrame, metadata_path: Path) -> Path:
    """Gera relatorio HTML autocontido com metricas, ICM e info de deployment."""
    print('\nGerando relatorio HTML...')

    agora = datetime.now().strftime('%d/%m/%Y %H:%M')

    # --- metricas ---
    mAP50    = metricas.get('mAP50',     'N/A')
    mAP5095  = metricas.get('mAP50_95',  'N/A')
    prec     = metricas.get('precision', 'N/A')
    rec      = metricas.get('recall',    'N/A')

    def fmt(v):
        return f'{v:.4f}' if isinstance(v, float) else str(v)

    # --- tabela ICM ---
    if not icm_df.empty:
        icm_media = icm_df['icm_score'].mean()
        icm_min   = icm_df['icm_score'].min()
        icm_max   = icm_df['icm_score'].max()

        icm_rows = ''.join(
            f'<tr><td>{r.imagem}</td><td>{r.icm_score:.1f}</td>'
            f'<td><span class="badge badge-{r.condicao.lower()}">{r.condicao}</span></td>'
            f'<td>{r.n_deteccoes}</td></tr>'
            for r in icm_df.itertuples()
        )
        icm_grafico_b64 = _grafico_icm_base64(icm_df)
        icm_img_tag = f'<img src="data:image/png;base64,{icm_grafico_b64}" style="max-width:100%;border-radius:8px;">'
        icm_summary = f'''
        <div class="stat-row">
          <div class="stat-card"><div class="stat-value">{icm_media:.1f}</div><div class="stat-label">ICM Medio</div></div>
          <div class="stat-card"><div class="stat-value">{icm_min:.1f}</div><div class="stat-label">ICM Minimo</div></div>
          <div class="stat-card"><div class="stat-value">{icm_max:.1f}</div><div class="stat-label">ICM Maximo</div></div>
          <div class="stat-card"><div class="stat-value">{len(icm_df)}</div><div class="stat-label">Imagens Avaliadas</div></div>
        </div>'''
    else:
        icm_rows = '<tr><td colspan="4" style="text-align:center;color:#888;">Sem dados de ICM</td></tr>'
        icm_img_tag = ''
        icm_summary = ''

    # --- tabela de classes + pesos ICM ---
    classes_rows = ''.join(
        f'<tr><td>{i}</td><td>{name}</td>'
        f'<td>{config.TEXT_PROMPTS.get(name, "-")}</td>'
        f'<td>{config.ICM_PESO.get(name, 1)}</td></tr>'
        for i, name in enumerate(config.CLASS_NAMES)
    )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatorio Final — Pavimentos YOLOv8</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#222;padding:24px}}
  h1{{font-size:1.8rem;font-weight:700;color:#1a1a2e}}
  h2{{font-size:1.15rem;font-weight:600;color:#16213e;margin:28px 0 12px;border-left:4px solid #4361ee;padding-left:10px}}
  .header{{background:linear-gradient(135deg,#4361ee,#3a0ca3);color:#fff;border-radius:12px;padding:28px 32px;margin-bottom:28px}}
  .header p{{opacity:.85;font-size:.95rem;margin-top:6px}}
  .card{{background:#fff;border-radius:10px;padding:20px 24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  .stat-row{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}}
  .stat-card{{flex:1;min-width:120px;background:#f8f9ff;border:1px solid #e0e4ff;border-radius:8px;padding:14px 16px;text-align:center}}
  .stat-value{{font-size:1.6rem;font-weight:700;color:#4361ee}}
  .stat-label{{font-size:.78rem;color:#666;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  th{{background:#4361ee;color:#fff;padding:9px 12px;text-align:left;font-weight:600}}
  td{{padding:8px 12px;border-bottom:1px solid #eee}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#f5f6ff}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.78rem;font-weight:600;color:#fff}}
  .badge-otimo{{background:#2ecc71}}.badge-bom{{background:#27ae60}}
  .badge-regular{{background:#f39c12}}.badge-ruim{{background:#e67e22}}
  .badge-pessimo{{background:#e74c3c}}
  .file-list{{list-style:none;padding:0}}
  .file-list li{{padding:7px 0;border-bottom:1px solid #eee;font-size:.9rem;display:flex;align-items:center;gap:8px}}
  .file-list li:last-child{{border-bottom:none}}
  .tag{{display:inline-block;background:#e8edff;color:#4361ee;border-radius:4px;padding:2px 8px;font-size:.75rem;font-weight:600}}
  footer{{text-align:center;color:#aaa;font-size:.8rem;margin-top:32px}}
</style>
</head>
<body>

<div class="header">
  <h1>&#x1F6E3; Pavimentos Brasil — Relatorio Final YOLOv8</h1>
  <p>Pipeline de deteccao de defeitos em pavimentos urbanos &nbsp;|&nbsp; Gerado em {agora}</p>
</div>

<!-- Metricas -->
<div class="card">
  <h2>Metricas de Desempenho (conjunto de teste)</h2>
  <div class="stat-row">
    <div class="stat-card"><div class="stat-value">{fmt(mAP50)}</div><div class="stat-label">mAP@0.50</div></div>
    <div class="stat-card"><div class="stat-value">{fmt(mAP5095)}</div><div class="stat-label">mAP@0.50:0.95</div></div>
    <div class="stat-card"><div class="stat-value">{fmt(prec)}</div><div class="stat-label">Precision</div></div>
    <div class="stat-card"><div class="stat-value">{fmt(rec)}</div><div class="stat-label">Recall</div></div>
  </div>
</div>

<!-- Config do modelo -->
<div class="card">
  <h2>Configuracao do Modelo</h2>
  <div class="stat-row">
    <div class="stat-card"><div class="stat-value">{config.YOLO_MODEL}</div><div class="stat-label">Arquitetura</div></div>
    <div class="stat-card"><div class="stat-value">{config.IMG_SIZE}px</div><div class="stat-label">Tamanho de entrada</div></div>
    <div class="stat-card"><div class="stat-value">{config.EPOCHS}</div><div class="stat-label">Epocas</div></div>
    <div class="stat-card"><div class="stat-value">{config.CONF_THRESH}</div><div class="stat-label">Conf. Threshold</div></div>
    <div class="stat-card"><div class="stat-value">{config.IOU_THRESH}</div><div class="stat-label">IoU Threshold</div></div>
  </div>
</div>

<!-- Classes -->
<div class="card">
  <h2>Classes Detectadas</h2>
  <table>
    <thead><tr><th>#</th><th>Classe</th><th>Prompt (Grounding DINO)</th><th>Peso ICM</th></tr></thead>
    <tbody>{classes_rows}</tbody>
  </table>
</div>

<!-- ICM -->
<div class="card">
  <h2>Indice de Condicao de Manutencao (ICM)</h2>
  {icm_summary}
  {icm_img_tag}
  <br>
  <table>
    <thead><tr><th>Imagem</th><th>ICM Score</th><th>Condicao</th><th>Deteccoes</th></tr></thead>
    <tbody>{icm_rows}</tbody>
  </table>
</div>

<!-- Arquivos exportados -->
<div class="card">
  <h2>Artefatos de Deployment</h2>
  <ul class="file-list">
    <li><span class="tag">ONNX</span> pavimentos_yolov8.onnx — deployment multiplataforma (OpenCV, ONNX Runtime, TensorRT)</li>
    <li><span class="tag">PT</span> pavimentos_yolov8_best.pt — pesos originais PyTorch (ultralytics)</li>
    <li><span class="tag">TS</span> pavimentos_yolov8_torchscript.torchscript — inferencia CPU/mobile otimizada</li>
    <li><span class="tag">JSON</span> model_metadata.json — metadados do modelo (classes, thresholds, metricas)</li>
    <li><span class="tag">ZIP</span> pacote_deploy.zip — todos os artefatos empacotados</li>
  </ul>
</div>

<!-- Como usar -->
<div class="card">
  <h2>Como Usar o Modelo (ONNX)</h2>
  <pre style="background:#1a1a2e;color:#e0e0e0;padding:16px;border-radius:8px;font-size:.82rem;overflow-x:auto;">
<span style="color:#7ec8e3">from</span> ultralytics <span style="color:#7ec8e3">import</span> YOLO
<span style="color:#7ec8e3">import</span> json, pathlib

<span style="color:#aaa"># Carrega metadados</span>
meta = json.loads(pathlib.Path(<span style="color:#f8b400">'model_metadata.json'</span>).read_text())
print(<span style="color:#f8b400">'Classes:'</span>, meta[<span style="color:#f8b400">'classes'</span>])
print(<span style="color:#f8b400">'mAP50:'</span>, meta.get(<span style="color:#f8b400">'mAP50'</span>, <span style="color:#f8b400">'N/A'</span>))

<span style="color:#aaa"># Inferencia com ONNX</span>
model = YOLO(<span style="color:#f8b400">'pavimentos_yolov8.onnx'</span>)
results = model.predict(<span style="color:#f8b400">'imagem.jpg'</span>, conf={config.CONF_THRESH}, iou={config.IOU_THRESH})
results[0].show()          <span style="color:#aaa"># visualizar</span>
results[0].save()          <span style="color:#aaa"># salvar imagem anotada</span></pre>
</div>

<footer>Pavimentos-Brasil YOLOv8 Pipeline &mdash; {agora}</footer>
</body>
</html>"""

    out = EXPORT_DIR / 'relatorio_final.html'
    out.write_text(html, encoding='utf-8')
    print(f'Relatorio salvo: {out}')
    return out


# ---------------------------------------------------------------------------
# Empacotamento ZIP
# ---------------------------------------------------------------------------

def empacotar_deploy(arquivos: list[Path]) -> Path:
    """Empacota todos os artefatos em um unico ZIP para entrega."""
    import zipfile

    zip_path = EXPORT_DIR / 'pacote_deploy.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in arquivos:
            if f and f.exists():
                zf.write(f, arcname=f.name)

    tamanho_mb = zip_path.stat().st_size / 1024 ** 2
    print(f'\nPacote ZIP criado: {zip_path}  ({tamanho_mb:.1f} MB)')
    return zip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('  05_export.py — Exportacao do Modelo Pavimentos YOLOv8')
    print('=' * 60)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    model, best_weights = carregar_modelo()
    metricas            = carregar_metricas()
    icm_df              = carregar_icm_csv()

    # --- Exportar formatos ---
    onnx_path = exportar_onnx(model, best_weights)
    ts_path   = exportar_torchscript(model, best_weights)
    pt_path   = copiar_pesos(best_weights)

    arquivos_exportados = {
        'onnx'         : onnx_path,
        'torchscript'  : ts_path,
        'pytorch_pt'   : pt_path,
    }

    # --- Metadata ---
    meta_path = salvar_metadata(metricas, arquivos_exportados)

    # --- Relatorio HTML ---
    html_path = gerar_relatorio_html(metricas, icm_df, meta_path)

    # --- ZIP ---
    todos = [onnx_path, ts_path, pt_path, meta_path, html_path]
    zip_path = empacotar_deploy(todos)

    # --- Resumo final ---
    print('\n' + '=' * 60)
    print('  Exportacao concluida!')
    print('=' * 60)
    print(f'\nArtefatos em: {EXPORT_DIR}\n')

    for label, path in [
        ('ONNX          ', onnx_path),
        ('TorchScript   ', ts_path),
        ('Pesos (.pt)   ', pt_path),
        ('Metadados     ', meta_path),
        ('Relatorio HTML', html_path),
        ('Pacote ZIP    ', zip_path),
    ]:
        status = '✓' if (path and Path(path).exists()) else '✗ nao gerado'
        print(f'  [{status}] {label}: {path}')

    print()


if __name__ == '__main__':
    main()
