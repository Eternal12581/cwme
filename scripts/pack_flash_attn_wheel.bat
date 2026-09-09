@echo off
setlocal EnableExtensions
REM Save wheel to D:\CWME. No generated .sh (avoids CRLF).
set IMG=davis-qwen35-9b:cuda121-pg
set OUT=D:\CWME
set WHL=flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
set URL1=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/%WHL%
set URL2=https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/%WHL%
if not exist %OUT% mkdir %OUT%

echo pack image=%IMG%
echo pack out=%OUT%
echo pack file=%WHL%
docker info >nul
if errorlevel 1 goto nodocker

echo try GitHub ...
docker run --rm -v D:/CWME:/wheels %IMG% wget -O /wheels/%WHL% %URL1%
if exist %OUT%\%WHL% goto check

echo try ghfast mirror ...
docker run --rm -v D:/CWME:/wheels %IMG% wget -O /wheels/%WHL% %URL2%
if exist %OUT%\%WHL% goto check

echo DOWNLOAD FAILED.
echo Open a browser and save this file into D:\CWME
echo %URL1%
echo or
echo %URL2%
exit /b 1

:check
for %%A in (%OUT%\%WHL%) do set SZ=%%~zA
echo size=%SZ%
if %SZ% LSS 1000000 (
  echo file too small, not a real wheel. delete and retry browser download.
  del %OUT%\%WHL%
  exit /b 1
)
echo pack OK: %OUT%\%WHL%
echo Upload this file to /mnt/workspace/wheels_flash_attn/
exit /b 0

:nodocker
echo Start Docker Desktop, then: docker info
exit /b 1
