@echo off
call E:\anaconda\Scripts\activate.bat Py
echo Installing onnxruntime in conda env "Py"...
pip install onnxruntime
echo Done.
pause
