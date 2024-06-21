#################### PRINT UNIQUE VALUES #################### 
def print_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Unique values in column '{column}': {unique_values}")
############################################################
