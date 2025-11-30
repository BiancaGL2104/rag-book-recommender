import pandas as pd 

def load_books(path="data/clean_books.csv"):
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset with {len(df)} books.")
        return df
    except FileNotFoundError:
        print("ERROR: clean_books.csv not found.")
        return None