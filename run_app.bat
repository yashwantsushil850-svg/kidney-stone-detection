@echo off

call C:\Users\Yashw\anaconda3\Scripts\activate.bat kidney_ai

cd /d C:\Users\Yashw\Downloads\KidneyStoneDetectionApp

python -m streamlit run app.py

pause