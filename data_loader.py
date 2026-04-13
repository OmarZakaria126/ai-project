import pandas as pd
from config import EXTERNAL_FILE, PREVIOUS_FILE

def load_datasets():
    external_df = pd.read_excel(EXTERNAL_FILE)
    previous_df = pd.read_excel(PREVIOUS_FILE)

    external_projects = external_df.to_dict(orient="records")
    previous_projects = previous_df.to_dict(orient="records")

    return external_projects, previous_projects
