@echo off
echo Creating conda environment "seg" with Python 3.10...
call E:\anaconda\Scripts\activate.bat
conda create -n seg python=3.10 -y
echo.
echo Installing dependencies...
call E:\anaconda\Scripts\activate.bat seg
pip install -r "%~dp0environment_requirement.txt"
echo.
echo Environment "seg" is ready. Activate with: conda activate seg
pause
