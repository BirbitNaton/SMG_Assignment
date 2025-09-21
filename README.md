# SMG_Assignment

This is the practical assignment for the Swiss Marketplace Group. The overall task is to demonstrate the stack using class Madrid Housing Market dataset from Kaggle.

# Installation

1. Clone the repoository to your machine.
2. Locate to the project directory.
3. Make sure `python 3.11` is installed, 3.11.9 recommended.
4. Make sure `pip 24` is installed .
5. Run `python -m venv venv` to create virtual environment.
6. Activate venv via:
    - **Unix system** - `source .venv/bin/activate`
    - **Windows system** - `./venv/Scripts/Activate.ps1`
7. Run `pip install -e .` to install required dependencies.

# Runtime
1. Create your settings.json in `./task1/model_settings/` or use one of the presets present there.
2. Train a model. There're 2(4) ways to do it. 
    - Use these comands to run with default parameters:
        - **Unix system** - run `python task1/training/main.py`
        - **Windows system** - run `python task1\training\main.py`
    - Use these commands to run with custom parameters:
        - **Unix system** - run `python task1/training/main.py --settings_json_path *settings_filename*.json`
        - **Windows system** - run `python task1\training\main.py --settings_json_path *settings_filename*.json`
    - If using VSCode, locate to `./.vscode/launch.json` and run either `run_training_default` or `run_training_custom` depending on whether you want to train on default or custom parameters set.

# Monitoring and post
- To monitor models and training process, run `mlflo ui` to run mlflow dashboard and locate to the local host as provided in the output. If default localhost is prompted, [follow this link](http://127.0.0.1:5000).
- to use POST terminal created with FastAPI, run `uvicorn task1.api.main:app --reload` and locate to the local host as provided in the output. If default localhost is prompted, [follow this link](http://127.0.0.1:8000/docs#).