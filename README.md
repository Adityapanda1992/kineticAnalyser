# EnzymeKinetic Analyzer

## Overview
This application performs robust non-linear regression on enzyme kinetics data to determine parameters like **Vmax**, **Km**, and **Ki**. It supports several models and diagnostic tools.

## Features
- **Multiple Models**: Michaelis-Menten, Substrate Inhibition, and various Inhibition models (Matrix input).
- **Weighting Schemes**: Supported OLS (ordinary least squares), Poisson (1/y), and Relative (1/y²) weighting for accurate error estimation.
- **Visual Diagnostics**: Generates Fit curves and Residual plots.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/kineticAnalyser.git
   cd kineticAnalyser
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   python enzyme_app.py
   ```

## Requirements
- Python 3.8+
- Flet
- NumPy
- SciPy
- Matplotlib

## Building Standalone Executable
you can build your own standalone executable for Windows, Linux, or macOS locally.

1. **Install Flet**:
   ```bash
   pip install flet
   ```

2. **Run the packaging command**:
   - **Windows**:
     ```bash
     flet pack enzyme_app.py -n "EnzymeKinetic Analyzer" -i icon.ico
     ```
   - **macOS/Linux**:
     ```bash
     flet pack enzyme_app.py -n "EnzymeKinetic Analyzer" -i icon.png
     ```

The executable will be generated in the `dist` folder.

## License
MIT License
