@echo off
echo Installing dependencies...
pip install -r requirements.txt -r requirements-dev.txt

echo.
echo Done! Try the offline mock pipeline:
echo     set PYTHONPATH=src ^&^& python -m model.synthesize --mock
echo Or, with make:  make all
pause
