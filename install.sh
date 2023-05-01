#!/bin/bash

# Display a message explaining what the script does
echo "This script will download and install AlignSeniorSem."
echo "Do you want to continue? [Y/n]"
read answer

# Check if the user wants to continue
if [[ ! $answer =~ ^[Yy]$ ]]; then
    echo "Aborting installation"
    exit 1
fi

# Check if the script is being run as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root" 
   exit 1
fi

# Check for Python installation and version
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Python is not installed"
    exit 1
fi

if command -v python &> /dev/null; then
    version=$(python -c 'import platform; print(platform.python_version())')
    venv_python="python"
elif command -v python3 &> /dev/null; then
    version=$(python3 -c 'import platform; print(platform.python_version())')
    venv_python="python3"
fi

if [[ $version != 3.8.* && $version != 3.9.* && $version != 3.10.* ]]; then
    echo "Python version $version is not supported. Please use Python 3.8, 3.9, or 3.10."
    exit 1
fi

# Clone the repository into the user's home directory
cd ~
git clone https://github.com/andreigerashchenko/AlignSeniorSem

# Create virtual environment, activate it, and install dependencies
cd AlignSeniorSem
$venv_python -m venv env
source env/bin/activate
pip install -r requirements_linux.txt

# Prompt the user to create a symbolic link to the program on their desktop
echo "Do you want to create a symbolic link to the program on your desktop? [Y/n]"
read symlink_answer

if [[ $symlink_answer =~ ^[Yy]$ ]]; then
    desktop_path="$HOME/Desktop/AlignSeniorSem.desktop"
    echo "[Desktop Entry]" > $desktop_path
    echo "Type=Application" >> $desktop_path
    echo "Name=AlignSeniorSem" >> $desktop_path
    echo "Exec=$HOME/AlignSeniorSem/run.sh" >> $desktop_path
    echo "Icon=$HOME/AlignSeniorSem/icon.png" >> $desktop_path
    echo "Terminal=false" >> $desktop_path
    chmod +x $desktop_path
    echo "A symbolic link to the program has been created on your desktop"
fi
