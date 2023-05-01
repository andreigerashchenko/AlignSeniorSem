#!/bin/bash

# Check if the script is being run as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root" 
   exit 1
fi

# Check if the virtual environment exists
if [ ! -d "$HOME/AlignSeniorSem/env" ]; then
    echo "Virtual environment not found. Please run the install script first."
    exit 1
fi

# Activate the virtual environment
source $HOME/AlignSeniorSem/env/bin/activate

# Parse the command-line arguments
show_console=true
while getopts ":v" opt; do
  case $opt in
    v)
      show_console=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Start the Python program with or without console based on the -v flag
if [ "$show_console" = true ]; then
    python $HOME/AlignSeniorSem/main_screen.py
else
    pythonw $HOME/AlignSeniorSem/main_screen.py
fi
