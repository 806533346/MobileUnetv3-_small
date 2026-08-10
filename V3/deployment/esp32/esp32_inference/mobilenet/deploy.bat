@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   ESP32-S3 分割模型一键部署
echo ========================================
echo.

REM ---- 检查 ESP-IDF 环境 ----
where esptool.py >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 esptool.py, 请先运行 ESP-IDF 导出脚本
    pause
    exit /b 1
)

REM ---- 端口 ----
set PORT=COM5
if not "%1"=="" set PORT=%1
echo 目标端口: %PORT%

echo.
echo [1/6] 生成测试数据...
python ..\..\generate_test_data.py
if %errorlevel% neq 0 (
    echo 测试数据生成失败!
    pause
    exit /b 1
)

echo.
echo [2/6] 打包 storage 分区镜像...
python pack_storage_bin.py
if %errorlevel% neq 0 (
    echo 打包失败!
    pause
    exit /b 1
)

echo.
echo [3/6] 检查模型文件...
if not exist "..\model\model.espdl" (
    echo model.espdl 不存在, 运行转换...
    python ..\..\convert_to_espdl.py
    if %errorlevel% neq 0 (
        echo 模型转换失败!
        pause
        exit /b 1
    )
) else (
    echo model.espdl 已存在, 跳过转换
)

echo.
echo [4/6] 编译项目...
call idf.py build
if %errorlevel% neq 0 (
    echo 编译失败!
    pause
    exit /b 1
)

echo.
echo [5/6] 烧录固件 + 模型 + 测试数据...
call idf.py -p %PORT% flash
if %errorlevel% neq 0 (
    echo 固件烧录失败!
    pause
    exit /b 1
)

esptool.py --chip esp32s3 -p %PORT% -b 460800 write-flash 0x410000 ..\model\model.espdl
if %errorlevel% neq 0 (
    echo 模型烧录失败!
    pause
    exit /b 1
)

esptool.py --chip esp32s3 -p %PORT% -b 460800 write-flash 0x610000 storage.bin
if %errorlevel% neq 0 (
    echo 测试数据烧录失败!
    pause
    exit /b 1
)

echo.
echo [6/6] 启动串口监控 (Ctrl+] 退出)...
call idf.py -p %PORT% monitor
