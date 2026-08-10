@echo off
chcp 65001 >nul
echo ============================================================
echo   ESP32 部署环境搭建脚本
echo ============================================================
echo.

:: ===== 基础配置 =====
set CONDA_BASE=E:\anaconda
set ENV_NAME=esp32
set PYTHON_VER=3.13

:: 检查 conda
if not exist "%CONDA_BASE%\Scripts\conda.exe" (
    echo [ERROR] 找不到 conda: %CONDA_BASE%
    echo 请修改 CONDA_BASE 为你的 Anaconda 安装路径
    pause
    exit /b 1
)

echo [1/4] 检查 conda 环境...
call "%CONDA_BASE%\Scripts\activate.bat" %CONDA_BASE%

conda env list | findstr /c:"%ENV_NAME%" >nul
if %errorlevel% neq 0 (
    echo   创建新环境: %ENV_NAME% (Python %PYTHON_VER%)
    conda create -n %ENV_NAME% python=%PYTHON_VER% -y
    if %errorlevel% neq 0 (
        echo [ERROR] conda 创建环境失败
        pause
        exit /b 1
    )
) else (
    echo   环境 %ENV_NAME% 已存在, 跳过创建
)

echo.
echo [2/4] 安装 PyTorch (CPU)...
call conda run -n %ENV_NAME% pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo [WARN] PyTorch 安装失败, 尝试默认源...
    call conda run -n %ENV_NAME% pip install torch torchvision
)

echo.
echo [3/4] 安装 ESP-PPQ...
call conda run -n %ENV_NAME% pip install esp-ppq
if %errorlevel% neq 0 (
    echo [WARN] esp-ppq 安装失败, 请手动安装
)

echo.
echo [4/4] 安装通用依赖...
call conda run -n %ENV_NAME% pip install opencv-python albumentations pyserial tqdm pyyaml pillow numpy
if %errorlevel% neq 0 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   安装完成!
echo ============================================================
echo.
echo 环境名称: %ENV_NAME%
echo 激活方式: conda activate %ENV_NAME%
echo.
echo 已安装包:
call conda run -n %ENV_NAME% pip list 2>nul | findstr /i "torch esp-ppq opencv albumentations pyserial tqdm pyyaml numpy"
echo.
echo 运行测试:
echo   conda activate %ENV_NAME%
echo   python _run_test.py                # ESP32 硬件推理测试
echo   python _validate_full_testset.py   # PC 端 INT8 仿真验证
echo   python _fp32_baseline.py           # FP32 基线测试
echo   python convert_to_espdl.py         # 模型量化导出
echo.
pause
