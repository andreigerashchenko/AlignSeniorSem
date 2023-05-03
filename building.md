# Building the project

### Windows

1. Install Python 3.10.11
2. Create a virtual environment named `env` in the project root directory
3. Activate the virtual environment
4. Install the required packages using `pip install -r build_requirements_windows.txt`
5. Run `pyinstaller Align.spec -y` from the project root directory
6. The executable and its accompanying files will be in the `dist` folder