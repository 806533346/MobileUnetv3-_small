@echo off
subst X: "C:\Users\匿名" >nul 2>nul
set IDF_PATH=C:\esp\v6.0.2\esp-idf
set ESP_IDF_VERSION=6.0
set IDF_PYTHON_ENV_PATH=X:\.espressif\python_env\idf6.0_py3.13_env
set IDF_CCACHE_ENABLE=1
set OPENOCD_SCRIPTS=X:\.espressif\tools\openocd-esp32\v0.12.0-esp32-20260424\openocd-esp32\share\openocd\scripts
set ESP_ROM_ELF_DIR=X:\.espressif\tools\esp-rom-elfs\20241011\
set IDF_TOOLS_PATH=X:\.espressif
set PATH=C:\esp\v6.0.2\esp-idf\components\espcoredump;C:\esp\v6.0.2\esp-idf\components\partition_table;C:\esp\v6.0.2\esp-idf\components\app_update;X:\.espressif\tools\xtensa-esp-elf-gdb\17.1_20260402\xtensa-esp-elf-gdb\bin;X:\.espressif\tools\riscv32-esp-elf-gdb\17.1_20260402\riscv32-esp-elf-gdb\bin;X:\.espressif\tools\xtensa-esp-elf\esp-15.2.0_20251204\xtensa-esp-elf\bin;X:\.espressif\tools\riscv32-esp-elf\esp-15.2.0_20251204\riscv32-esp-elf\bin;X:\.espressif\tools\esp32ulp-elf\2.38_20240113\esp32ulp-elf\bin;X:\.espressif\tools\cmake\4.0.3\bin;X:\.espressif\tools\openocd-esp32\v0.12.0-esp32-20260424\openocd-esp32\bin;X:\.espressif\tools\ninja\1.12.1\;X:\.espressif\tools\idf-exe\1.0.3\;X:\.espressif\tools\ccache\4.12.1\ccache-4.12.1-windows-x86_64;X:\.espressif\tools\dfu-util\0.11\dfu-util-0.11-win64;X:\.espressif\python_env\idf6.0_py3.13_env\Scripts;C:\esp\v6.0.2\esp-idf\tools;%PATH%
cd /d D:\CODE\mobileunetv3++\MobileV3Unet++4\V3\deployment\esp32\esp32_inference\mobilenet
echo === Starting build ===
python "%IDF_PATH%\tools\idf.py" build
if %errorlevel% neq 0 (
    echo === BUILD FAILED ===
    pause
    exit /b 1
)
echo === BUILD SUCCESS ===
pause
