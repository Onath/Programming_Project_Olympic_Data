from sklearn.ensemble import RandomForestRegressor

#################### PRINT UNIQUE VALUES #####################
def print_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Unique values in column '{column}': {unique_values}")
##############################################################


########## IMPUTE MISSING VALUES WITH RANDOM FOREST ########## 
def impute_missing_values_with_random_forest(df, column_name):
    # Select only numerical columns (excluding the target column)
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if column_name in numerical_columns:
        numerical_columns.remove(column_name)

    # Exclude columns with NaN values
    numerical_columns = [col for col in numerical_columns if not df[col].isna().any()]

    # Splitting the data into two parts: one where target column is missing and one where it's not
    df_with_target = df[df[column_name].notna()] #the target column is not NaN: These rows provide 
                        # valuable information because they contain actual data that the model can learn from
    df_without_target = df[df[column_name].isna()] #rows where the target column is NaN and needs to be imputed 
                                # (using the information from the rows where the target column is not empty.)

    # Prepare the features (X) and target (y) using only numerical columns
    x = df_with_target[numerical_columns]
    y = df_with_target[column_name]

    # Create and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(x, y)

    # Predicting the missing values
    predicted_values = model.predict(df_without_target[numerical_columns])

    # Fill in the missing values in the original DataFrame
    df.loc[df[column_name].isna(), column_name] = predicted_values
    return df