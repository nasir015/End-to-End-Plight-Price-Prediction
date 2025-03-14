import pandas as pd
import numpy as np




def export_to_csv(data, file_name):
    data.to_csv(file_name, index=False)
    print(f"Data exported to {file_name}")