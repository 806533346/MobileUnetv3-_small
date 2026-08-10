Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ESP32-S3 分割模型部署" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "[1/4] 编译..." -ForegroundColor Yellow
idf.py build
if ($LASTEXITCODE -ne 0) {
    Write-Host "编译失败!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "[2/4] 烧录固件..." -ForegroundColor Yellow
idf.py flash
if ($LASTEXITCODE -ne 0) {
    Write-Host "烧录失败!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "[3/4] 烧录模型..." -ForegroundColor Yellow
esptool.py --chip esp32s3 -p COM5 -b 460800 write-flash 0x410000 ..\model\model.espdl
if ($LASTEXITCODE -ne 0) {
    Write-Host "模型烧录失败!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "[4/4] 串口监控 (Ctrl+] 退出)..." -ForegroundColor Yellow
idf.py monitor
