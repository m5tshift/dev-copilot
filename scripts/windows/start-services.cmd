@echo off
rem Brings the PC stand back after a reboot. Run in the interactive session:
rem   powershell -ExecutionPolicy Bypass -File scripts\windows\interactive.ps1 C:\src\devcopilot\scripts\windows\start-services.cmd
rem Ollama starts by itself (its own autostart). Log: C:\src\services.log
chcp 65001 >nul
set LOG=C:\src\services.log
echo === %DATE% %TIME% > %LOG%
rem 1. Docker Desktop (user install), then wait for the engine.
start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
:docker
docker version --format "{{.Server.Version}}" >> %LOG% 2>&1 || (powershell -NoProfile -Command "Start-Sleep 5" & goto docker)
rem 2. Observability stack. Ports and the Telegram chat id: observability\.env.
cd /d %~dp0..\..\observability
docker compose up -d >> %LOG% 2>&1
rem 3. sbx daemon with TMPDIR: without it mkfs.erofs fails with "No space left on device".
set TMPDIR=C:\src\sbxtmp
set TEMP=C:\src\sbxtmp
set TMP=C:\src\sbxtmp
if not exist C:\src\sbxtmp mkdir C:\src\sbxtmp
sbx daemon restart >> %LOG% 2>&1
sbx ls >> %LOG% 2>&1
echo === ready %TIME% >> %LOG%
