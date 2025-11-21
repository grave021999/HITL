# Python Setup Guide for Windows

## Issue
Python is not installed on your system. The `python` and `python3` commands are pointing to Microsoft Store stubs that require installation.

## Solution: Install Python

### Option 1: Install from Python.org (Recommended)

1. **Download Python:**
   - Go to https://www.python.org/downloads/
   - Download Python 3.11 or 3.12 (latest stable version)

2. **Install Python:**
   - Run the installer
   - **IMPORTANT:** Check the box "Add Python to PATH" during installation
   - Click "Install Now"

3. **Verify Installation:**
   - Open a new PowerShell window
   - Run: `python --version`
   - You should see something like: `Python 3.12.x`

4. **Install Dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

### Option 2: Install via Microsoft Store

1. Open Microsoft Store
2. Search for "Python 3.12" or "Python 3.11"
3. Click "Install"
4. After installation, open a new PowerShell window
5. Run: `python -m pip install -r requirements.txt`

### Option 3: Use Anaconda/Miniconda

1. Download Anaconda from https://www.anaconda.com/download
2. Install it
3. Open Anaconda Prompt
4. Navigate to project directory
5. Run: `pip install -r requirements.txt`

## After Installation

Once Python is installed:

1. **Close and reopen your PowerShell window** (important for PATH to update)

2. **Verify Python works:**
   ```powershell
   python --version
   pip --version
   ```

3. **Install project dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Run the test:**
   ```powershell
   python hitl_test.py
   ```

## Troubleshooting

If `pip` is still not recognized after installing Python:
- Make sure you checked "Add Python to PATH" during installation
- Restart your PowerShell/terminal window
- Try: `python -m pip` instead of just `pip`
- Check Python installation: `python -m pip --version`


