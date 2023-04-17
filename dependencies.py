# Installs OS-specific dependencies

import os
import sys

def main():
    os_name = sys.platform

    if os_name == 'win32': # Windows
        os.system('pip install -r requirements_windows.txt')
    elif os_name == 'linux': # Linux
        os.system('pip install -r requirements_linux.txt')
    elif os_name == 'darwin': # macOS
        os.system('pip install -r requirements_macos.txt')
    else: # Unsupported OS
        print('Unsupported OS')

if __name__ == '__main__':
    main()