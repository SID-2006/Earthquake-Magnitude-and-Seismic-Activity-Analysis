@echo off
title TerraPulse AI - Earthquake Seismic Intelligence Suite
echo =====================================================================
echo  TerraPulse AI - Global Seismic Intelligence & Magnitude Classifier
echo =====================================================================
echo.
echo Launching Streamlit application on http://localhost:8502 ...
echo.

py -3.14 -m streamlit run app.py --server.port 8502 --browser.gatherUsageStats false

pause
