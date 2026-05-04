
rmdir /s /q build dist
del *.spec


pyinstaller --windowed ^
    --name "SpikingNN_Simulator" ^
    --paths "." ^
	--icon=artificial-intelligence.ico ^
    --hidden-import matplotlib.backends.backend_tkagg ^
    --hidden-import SpikingNN.Izh_net ^
    --exclude-module PyQt5 ^
    --exclude-module PySide2 ^
    --exclude-module PySide6 ^
    --collect-all matplotlib ^
    qtapp.py