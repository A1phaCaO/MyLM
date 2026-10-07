@echo off
rem Usage: launch_tensorboard.bat [ip] [port]   default: 0.0.0.0 6006
cd /d "%~dp0"
if "%~1"=="" (set "TB_HOST=0.0.0.0") else (set "TB_HOST=%~1")
if "%~2"=="" (set "TB_PORT=6006") else (set "TB_PORT=%~2")
echo serving on http://%TB_HOST%:%TB_PORT%/  (local: http://127.0.0.1:%TB_PORT%/)
uv run tensorboard --logdir logs --host=%TB_HOST% --port=%TB_PORT%
