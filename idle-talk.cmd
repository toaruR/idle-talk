@echo off
rem idle-talk start / stop / restart / status
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
python -m idle_talk.ctl %*
