@echo off
REM === (optionnel) activer l'environnement virtuel s'il existe ===
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate

REM === lancer l'application ===
streamlit run Kayak_NextStop.py

REM === garder la fenêtre ouverte si erreur ===
pause
