from sklearn.ensemble import RandomForestRegressor
import streamlit as st 

#################### PRINT UNIQUE VALUES #####################
def print_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Unique values in column '{column}': {unique_values}")
##############################################################


########## IMPUTE MISSING VALUES WITH RANDOM FOREST ########## 
def impute_missing_values_with_random_forest(df, column_name):
    
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if column_name in numerical_columns:
        numerical_columns.remove(column_name)

    numerical_columns = [col for col in numerical_columns if not df[col].isna().any()]

    df_with_target = df[df[column_name].notna()] 
    df_without_target = df[df[column_name].isna()] 

    # Prepare the features (X) and target (y) using only numerical columns
    x = df_with_target[numerical_columns]
    y = df_with_target[column_name]

    # Create and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(x, y)

    predicted_values = model.predict(df_without_target[numerical_columns])

    # Fill in the missing values in the original DataFrame
    df.loc[df[column_name].isna(), column_name] = predicted_values
    return df


##############################################################
def get_missing_values_info(df):
    missing_values = df.isna().sum()
    total_missing = missing_values.sum()
    missing_columns = missing_values[missing_values > 0].index.tolist()
    return total_missing, missing_columns


def display_missing_values_info(total_missing, missing_columns, df_name):
    if total_missing > 0:
        message = f"In {df_name}, there are {total_missing} missing values in columns: {', '.join(missing_columns)}"
    else:
        message = f"There are no missing values in {df_name}."
    
    st.markdown(f"""
    <div style="border:2px solid #d3d3d3; padding: 10px; border-radius: 5px;">
        {message}
    </div>
    """, unsafe_allow_html=True)
##############################################################