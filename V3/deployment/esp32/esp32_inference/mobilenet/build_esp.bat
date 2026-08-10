@echo off
set IDF_PATH=C:\esp\v6.0.2\esp-idf
set IDF_PYTHON_ENV_PATH=C:\Users\匿名\.espressif\python_env\idf6.0_py3.13_env
set ESP_IDF_VERSION=6.0
set IDF_TOOLS_PATH=C:\Users\匿名\.espressif
set PATH=C:\esp\v6.0.2\esp-idf\tools;C:\Users\匿名\.espressif\python_env\idf6.0_py3.13_env\Scripts;C:\Users\匿名\.espressif\tools\cmake\4.0.3\bin;C:\Users\匿名\.espressif\tools\ninja\1.12.1;C:\Users\匿名\.espressif\tools\xtensa-esp-elf\esp-15.2.0_20251204\xtensa-esp-elf\bin;C:\Users\匿名\.espressif\tools\idf-exe\1.0.2;C:\Users\匿名\.espressif\tools\ccache\4.11.1;%PATH%
cd /d D:\CODE\mobileunetv3++\MobileV3Unet++4\V3\deployment\esp32\esp32_inference\mobilenet
echo === Starting build ===
idf.py build
if %errorlevel% neq 0 (
    echo === BUILD FAILED ===
    pause
    exit /b 1
)
echo === BUILD SUCCESS ===
