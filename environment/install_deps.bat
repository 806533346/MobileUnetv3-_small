@echo off
call E:\anaconda\Scripts\activate.bat Py
echo Installing dependencies in conda env "Py"...
pip install -r "%~dp0environment_requirement.txt"
echo Done.
pause
