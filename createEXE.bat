@echo off
pyinstaller main.py --noconfirm --log-level=WARN ^
    --onedir --clean ^
    --specpath=".\\app\\" ^
    --workpath=".\\app\\build\\" ^
    --distpath=".\\app\\" ^
    --name="imageSim" ^
    --onefile --nowindow ^
    --icon="..\\images\\icon.ico" ^
