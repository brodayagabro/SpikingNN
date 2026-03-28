# SpikingNN Simulator - Build Instructions

## Important: Path Requirements

PyInstaller does not work with non-ASCII characters in paths.

**Do NOT use:** `C:\Users\Name\Рабочий стол\Project\`  
**Use instead:** `C:\Projects\SpikingNN\`

Ensure your project path contains only:
- Latin letters (a-z, A-Z)
- Numbers (0-9)
- Underscores (_) or hyphens (-)
- No spaces

---

## Requirements

- **Python 3.8 or higher**
- **Dependencies:** `numpy`, `matplotlib`, `networkx`, `pyinstaller`

---

## Build Instructions

### 1. Prepare the Environment

```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib networkx pyinstaller
# install SpikingNN library
cd SpikingNN
pip install .
```

### 2. Run the Build Script

```cmd
build_app.bat
```

### 3. Test the Application

```cmd
cd dist\SpikingNN_Simulator
SpikingNN_Simulator.exe
```

---

## Distribution

**Copy the entire folder from `dist`:**

```
dist/
└── SpikingNN_Simulator/      <-- Copy this entire folder
    ├── SpikingNN_Simulator.exe
    ├── _internal/            <-- Required!
    └── ...
```

**Create archive for distribution:**
```cmd
powershell Compress-Archive -Path "dist\SpikingNN_Simulator" -DestinationPath "SpikingNN_Simulator.zip"
```

**On target computer:**
1. Extract the ZIP archive
2. Run `SpikingNN_Simulator.exe`
3. Python is NOT required on the target machine

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Failed to load Python DLL` | Move project to a path without Cyrillic characters or spaces |
| `Qt plugin directory does not exist` | Ensure `matplotlib.use('TkAgg')` is at the top of main.py and PyQt5 is excluded |
| `ModuleNotFoundError: SpikingNN` | Verify `SpikingNN/__init__.py` exists and add `--hidden-import SpikingNN.Izh_net` |
| Icon not displaying | Delete `build/`, `dist/`, `*.spec` and rebuild |
| Application closes immediately | Remove `--windowed` flag temporarily to see console errors |

---

## Project Structure

```
SpikingNN_Simulator/
├── main.py                 # Main application file
├── SpikingNN/              # Neural network module
│   ├── __init__.py
│   └── Izh_net.py
├── icon.ico                # Application icon
├── build_app.bat           # Build script
├── requirements.txt        # Python dependencies
├── build/                  # Temporary build files (do not distribute)
├── dist/                   # Final application (distribute this)
└── README.md               # This file
```

---

## Notes

- The executable size will be approximately 400-600 MB due to numpy and matplotlib
- The application only works on Windows (for this build)
- Always test on a clean machine without Python installed before distribution

---

**Build complete. Ready for distribution.**
