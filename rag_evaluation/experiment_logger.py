import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_EXPERIMENTS_PATH = os.path.join(PROJECT_ROOT, "experiments.csv")

def log_experiment(results, filename=DEFAULT_EXPERIMENTS_PATH):

    file_exists = os.path.isfile(filename)

    df = pd.DataFrame([results])

    df.to_csv(
        filename,
        mode="a",
        header=not file_exists,
        index=False
    )
