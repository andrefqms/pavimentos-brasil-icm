@echo off
echo ============================================================
echo  Pavimentos-Brasil YOLOv8 - Setup Local
echo  Python 3.12 + CUDA 12.1 (RTX 3060)
echo ============================================================

echo Instalando PyTorch com CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Instalando demais dependencias...
pip install -r requirements.txt

echo.
echo Criando estrutura de pastas...
mkdir C:\pavimentos\data\raw 2>nul
mkdir C:\pavimentos\data\processed\yolo_dataset 2>nul
mkdir C:\pavimentos\models 2>nul
mkdir C:\pavimentos\results 2>nul

echo.
echo Verificando GPU...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma')"

echo.
echo ============================================================
echo  Setup concluido!
echo.
echo  Coloque as imagens em: C:\pavimentos\data\raw
echo  Ajuste os caminhos em config.py se necessario
echo.
echo  Ordem de execucao:
echo    1. python 01_pseudo_labels.py
echo    2. python 02_preparar_dataset.py
echo    3. python 03_treinar.py
echo    4. python 04_avaliar.py
echo ============================================================
pause
