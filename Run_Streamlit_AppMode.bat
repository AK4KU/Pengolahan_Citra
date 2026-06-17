@echo off
title Streamlit App Mode Launcher
echo =========================================================
echo  🎯 FPS Aim Performance Analyzer - Streamlit Dashboard
echo =========================================================
echo  Starting Streamlit server in headless background...
start /b "" "C:\laragon\bin\python\python-3.10\python.exe" -m streamlit run app.py --server.headless true
echo  Waiting 4 seconds for server boot...
timeout /t 4 /nobreak > nul
echo  Launching Chrome in Standalone Desktop Mode...
start chrome --app=http://localhost:8501
exit
