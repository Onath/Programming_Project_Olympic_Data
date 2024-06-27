import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functions as f
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
import io
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:,.2f}'.format 

# Load data
olympic_df = pd.read_csv('dataset_olympic_data/dataset_olympics.csv')
noc_region_df = pd.read_csv('dataset_olympic_data/noc_region.csv')


st.title("OLYMPIC DATA ANALYSIS")

############### INTRODUCTION ###############
st.header("Introduction")
st.markdown("""
            In this project, I analyzed two datasets:
            - Information on 70,000 Olympic athletes, including personal attributes, team affiliations, events, and medals won.
            - Details on 230 National Olympic Committees (NOCs) and their regions.

            My analysis focused on visualizing athlete participation by gender over the years, comparing total medals won by male and female athletes, examining age distribution, identifying top countries and athletes with the most medals, and analyzing the top 10 sports with the highest number of participants.
            """)
# Dataset presentation
st.write("## DATASET OVERVIEW")
st.write("Dataset [link](https://www.kaggle.com/datasets/bhanupratapbiswas/olympic-data/data)")

with st.expander("Dataset columns details"):
    features = [
        "**COLUMNS**",
        "1. **ID:** Identifier for each athlete.",
        "2. **NAME:** The full name of the athlete.",
        "3. **SEX:** The gender of the athlete, represented as 'M' for male and 'F' for female.",
        "4. **AGE:** The age of the athlete at the time of the Olympics.",
        "5. **HEIGHT:** The height of the athlete in centimeters.",
        "6. **WEIGHT:** The weight of the athlete in kilograms.",
        "7. **TEAM:** The country the athlete represents.",
        "8. **NOC:** The National Olympic Committee (NOC) code for the country the athlete represents.",
        "9. **GAMES:** The edition of the Olympics the athlete participated in, including the year and the season (Summer or Winter).",
        "10. **YEAR:** The year of the Olympics.",
        "11. **SEASON:** The season of the Olympics, either Summer or Winter.",
        "12. **CITY:** The host city of the Olympics.",
        "13. **SPORT:** The sport the athlete competed in.",
        "14. **EVENT:** The specific event within the sport that the athlete competed in.",
        "15. **MEDAL:** The type of medal won by the athlete, if any (Gold, Silver, Bronze, or NaN if no medal was won)."
    ]
    
    for feature in features:
        st.write(feature)

############### DATA EXPLORATION ###############
# elif option == "Data Exploration":
st.header("DATA EXPLORATION")

# Explore olympic_df
dataset = st.selectbox("Select Dataset to see more information", ["Olympic Data", "NOC Region Data"])
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

if dataset == "Olympic Data":
    st.subheader("Explore olympic_df")
    st.dataframe(olympic_df.head())

    buffer_olympic = io.StringIO()
    olympic_df.info(buf=buffer_olympic)
    info_olympic = buffer_olympic.getvalue()
    st.text(info_olympic)

    st.dataframe(olympic_df.describe())
    total_missing, missing_columns = get_missing_values_info(olympic_df)
    display_missing_values_info(total_missing, missing_columns, "olympic_df")

elif dataset == "NOC Region Data":
    st.subheader("Explore noc_region_df")
    st.dataframe(noc_region_df.head())

    buffer_noc = io.StringIO()
    noc_region_df.info(buf=buffer_noc)
    info_noc = buffer_noc.getvalue()
    st.text(info_noc)

    st.dataframe(noc_region_df.describe())
    total_missing, missing_columns = get_missing_values_info(noc_region_df)
    display_missing_values_info(total_missing, missing_columns, "noc_region_df")

# Olympic data  histogram
st.subheader("Olympic data histogram")
st.markdown("""From the histogram below we can see that:
            - The majority of the athletes age is between ~18 and ~25
            - Most athletes are between 160cm and 190cm 
            - Most athletes weigh between 50 and 100 and the peak is  around 175
            - Participation to the olympic games has increased over time. 
            """)
num_cols = len(olympic_df.select_dtypes(include='number').columns)
fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 6, 6))
for i, column in enumerate(olympic_df.select_dtypes(include='number').columns):
    olympic_df[column].hist(ax=axes[i], bins=10)
    axes[i].set_title(f'Histogram of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

plt.tight_layout()
st.pyplot(fig)

# Merge datasets
st.subheader("Merge NOC dataset with Olympic dataset")
st.markdown("""
            For convenience, I merged the two datasets. Although the *noc_region.csv* file does not contain information directly useful for future analysis, 
            I included the region column to easily identify the corresponding country for each NOC code.<br>
            Additionally, I renamed the *noc_region* column to NOC to facilitate merging the datasets on this common column.<br>
            In the *noc_region.csv* file, there is a column named notes that provides additional information about countries. However, 
            this information is not relevant to our analysis, and the column contains many missing values that we cannot fill. 
            Therefore, I will drop the notes column from dataset. 
            """)
noc_region_df.rename(columns={'noc_region' : 'NOC'}, inplace=True)
olympic_df = olympic_df.merge(noc_region_df, on='NOC')
olympic_df = olympic_df.drop(columns=['notes'])


############### DATA CLEANING ###############

st.header("Data Cleaning and Handling missing values")
st.markdown("""
            After merging the datasets, I will now work with *dataset_olympics.csv*, which I called *olympic_df*.
            
            First, I checked for and removed any duplicate values. 
            
            Before modifying the *medal* column, I created a binary column called *Athlete_won_medal*. 
            This column will be used in the prediction model I will develop later. 
            It contains '1' if the athlete won a medal (regardless of whether it is Gold, Silver, or Bronze) and '0' if the athlete did not win a medal. 
            In the *medal* column NaN values indicate that the athlete did not win a medal, so I will replace these NaN values with the string "No medal" using pandas' fillna() method.
            
            Additionally, I dropped rows with NaN values in the *region* column, as it is unclear which country or region they belong to.
            """)
# Check and drop duplicated rows
st.subheader("Check and drop duplicated rows")
duplicate_count = olympic_df.duplicated().sum()
st.write('There are {} duplicate rows in olympic_df based on all columns, before dropping them.'.format(duplicate_count))
olympic_df.drop_duplicates(keep='first', inplace=True)

# Binary column creation
olympic_df['Athlete Won Medal'] = olympic_df['Medal'].notnull().astype(int)

st.write("## Missing values")
olympic_df['Medal'] = olympic_df['Medal'].fillna(value='No medal')
olympic_df.dropna(subset=['reg'], inplace=True)

total_missing, missing_columns = get_missing_values_info(olympic_df)
display_missing_values_info(total_missing, missing_columns, "olympic_df")

# Fill missing values
st.markdown("""
            **Filling  missing values with random forest** 
            
            After cleaning the dataset, I filled the missing values using a Random Forest Regressor.

            ***Random Forest Regressor:***
            
            A Random Forest Regressor is an ensemble learning method that combines multiple decision trees to make predictions. 
            Each decision tree in the forest is trained on a random subset of the data, 
            and the final prediction is obtained by averaging the predictions of all the individual trees. 
            This approach helps improve the accuracy and robustness of the model by reducing the risk of overfitting and capturing complex patterns in the data.
            """)

for column in olympic_df.columns:
    if olympic_df[column].isna().any():
        olympic_df = f.impute_missing_values_with_random_forest(olympic_df, column)
        
    
############### DATA VISUALIZATION ###############

# GENDER DISTRIBUTION OVER THE YEARS
st.header("Data Visualization")
st.subheader("GENDER DISTRIBUTION OVER THE YEARS")
st.markdown("""It is evident that women's involvement was minimal before the 1970s, with a significant increase starting in the 1980s. 
            From this point onwards, the gap between the number of male atheletes and female atheletes began to decrease.
            """)
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=olympic_df, x="Year", hue='Sex', ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# MEDAL DISTRIBUTION OVER THE GENDER
st.subheader("MEDAL DISTRIBUTION OVER THE GENDER")
st.markdown("""We observe that, overall, men have won more medals than women. This is primarily due to the historically higher participation rates of men in the Olympic Games. However, starting from the 1970s and 1980s, the participation of women in the Olympics has significantly increased.""")
no_medal_mask= olympic_df[olympic_df['Medal']!='No medal']
colors = ['gold', 'darkgoldenrod', 'silver']

ax = sns.countplot(no_medal_mask, x="Sex", hue='Medal', palette=colors)
for container in ax.containers:
    ax.bar_label(container, label_type='center', rotation=0, fontsize=10)
plt.show()
st.pyplot(fig)

st.markdown("""To illustrate this trend, I plotted the medals won by male and female athletes after 1980. The data shows a narrowing gap between the number of medals won by men and women, confirming the higher participation of women in the Olympic Games in recent decades.
            """)
no_medal_mask = olympic_df[olympic_df['Medal']!='No medal']
df_after_1980 = no_medal_mask[no_medal_mask['Year']>1980].copy()

colors = [ 'darkgoldenrod', 'gold','silver']
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(df_after_1980, x="Sex", hue='Medal', palette=colors)
for container in ax.containers:
    ax.bar_label(container, label_type='center', rotation=0, fontsize=6)

st.pyplot(fig)

# ATHLETES AGE DISTRIBUTION
st.subheader("ATHLETES AGE DISTRIBUTION")
st.markdown("""**ADD COMMENT**""")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.histplot(olympic_df['Age'].values, kde=True, color='green', ax=ax)
# ax.set_title('Age distribution of Athletes', fontsize=15, fontweight='bold')
# ax.set_xlabel('Age')
# ax.set_ylabel('Frequency')

st.pyplot(fig)

# Top 10 team with most medals
st.subheader("Total medals gained by each country and visualization of top 10 countries with most medals")
st.markdown("""From the table, we can see that the United States leads with the highest number of total medals (1561) and the highest proportion of medals won (0.58). This indicates a high success rate for American athletes. Even if Italy has fewer medals than France, it has a slightly higher proportion (0.31) compared to France (0.27), suggesting that Italy had a more efficient medal-winning performance relative to the number of athletes.
            Great Britain, Germany, Australia, Sweden, and Canada also have notable performances, with varying proportions of medals won. Germany, for instance, has a proportion of 0.44, which is relatively high compared to its total number of athletes (982).
            """)
medal_count = olympic_df.groupby(['NOC','Medal']).size().unstack(fill_value=0).reset_index()
medal_count['Total'] = medal_count[['Gold', 'Silver', 'Bronze']].sum(axis=1)

ten_team_with_most_medals = medal_count.sort_values(by=['Total'], ascending=False).head(10)
team_medal_count = olympic_df.groupby(['NOC', 'Medal']).size()
team_athlete_count = olympic_df.groupby('NOC')['ID'].nunique().sort_values(ascending=False).head(10)
team_athlete_count = team_athlete_count.reset_index()
team_athlete_count.columns = ['NOC', 'Total Athletes']
ten_team_with_most_medals = ten_team_with_most_medals.merge(team_athlete_count, on="NOC")
ten_team_with_most_medals['Proportion'] = ten_team_with_most_medals['Total']/ten_team_with_most_medals['Total Athletes']
ten_team_with_most_medals

colors = ['gold', 'silver', 'darkgoldenrod']
fig, ax = plt.subplots(figsize=(12, 8))
ten_team_with_most_medals.set_index('NOC')[['Gold', 'Silver', 'Bronze']].plot(kind='bar', stacked=True, color=colors, ax=ax)
ax.set_title('Medal Counts by Country')
ax.set_xlabel('Country')
ax.set_ylabel('Number of Medals')

for container in ax.containers:
    ax.bar_label(container, label_type='center', rotation=90, fontsize=7)
plt.show()
st.pyplot(fig)

# Visualization of 10 athlete with most medals
st.subheader("10 ATHLETES WITH MOST MEDALS")
st.markdown("""In the table is displayed the success percentage of the top 10 athletes who have won the most medals. 
            The success percentage is calculated as the proportion of medals won to the total number of games participated in, 
            without differentiating between the types of medals (gold, silver, or bronze). 
            Interestingly, some athletes, such as Natalie Anne Coughlin and Raymond Clarence "Ray" Ewry, have achieved a 100% success rate, 
            meaning they won a medal in every event they participated in, even though they have fewer total medals compared to other athletes in the list.""")

athlete_medal_count = olympic_df.groupby(['ID','Medal']).size().unstack(fill_value=0)
athlete_medal_count['Total'] = athlete_medal_count[['Gold', 'Silver', 'Bronze']].sum(axis=1)
athlete_medal_count = athlete_medal_count.reset_index()
athlete_medal_count = athlete_medal_count[['ID','Gold', 'Silver', 'Bronze', 'Total']]
athlete_medal_count.columns.name = None
athlete_name = olympic_df[['ID', 'Name']].drop_duplicates()
athlete_medal_count = athlete_medal_count.merge(athlete_name, on='ID')
ten_athlete_with_most_medals = athlete_medal_count.sort_values(by='Total', ascending=False).head(10)
athlete_participation_count = olympic_df['Name'].value_counts()
ten_athlete_with_most_medals = ten_athlete_with_most_medals.merge(athlete_participation_count ,on='Name')
ten_athlete_with_most_medals.rename(columns={'count':'Total Participation'}, inplace=True)
ten_athlete_with_most_medals['athlete_success_percentage'] = (ten_athlete_with_most_medals['Total']/ten_athlete_with_most_medals['Total Participation'])*100

colors = ['gold', 'silver', 'darkgoldenrod']

fig, ax = plt.subplots(figsize=(12, 8))
ten_athlete_with_most_medals.set_index('Name')[['Gold', 'Silver', 'Bronze']].plot(kind='bar', stacked=True, color=colors, ax=ax)
ax.set_title('Medal Counts by Athlete')
ax.set_xlabel('Athlete')
ax.set_ylabel('Number of Medals')

for container in ax.containers:
    ax.bar_label(container, label_type='center', rotation=90, fontsize=7)
plt.show()
st.pyplot(fig)

# 10 SPORTS BY PARTCIPATION NUMBERS
st.subheader("10 SPORTS BY PARTCIPATION NUMBERS AND THE LEADING COUNTRIES IN EACH SPORT")
st.markdown(""" In the graph in below it is showed the number of athletes participating in various sports. Athletics has the highest participation, with around 6000 athletes, indicating its popularity. Swimming, Rowing, and Football also have significant participation, each with approximately 2000-2500 athletes. Other sports like Cycling, Boxing, Shooting, Wrestling, Sailing, and Fencing have lower participation levels, with around 1000 athletes each. 
            """)

sport_athlete_count = olympic_df.groupby('Sport')['ID'].nunique().sort_values(ascending=False).head(10)
sport_athlete_count = sport_athlete_count.reset_index()
sport_athlete_count = sport_athlete_count.rename(columns={'ID' : 'Total Participation'})

base_color = 'blue'
num_shades = len(sport_athlete_count)
colors = [mcolors.to_rgba(base_color, alpha) for alpha in np.linspace(1, 0.3, num_shades)]

fig, ax = plt.subplots(figsize=(12, 8))
sport_athlete_count.plot(kind='bar', x='Sport', y='Total Participation', ax=ax, color=colors)
ax.set_title('Number of Athletes per Sport', fontsize=15, fontweight='bold')
plt.xticks(rotation=90)

st.pyplot(fig)

# CORRELATION MATRIX
st.subheader("CORRELATION MATRIX")
st.markdown("Numerical encoding before showing correlation matrix")
st.markdown(""" **I used *get_dummies* function of pandas to transforms categorical variables into binary (0 or 1) variables.** """)
olympic_df = pd.get_dummies(olympic_df, columns=['Medal'], dtype=int)

# Convert object columns into numerical columns for preparing data for model
st.markdown("Convert object columns into numerical columns. ")
label_encoder = LabelEncoder()
for column in ['Sex' ,'Games', 'Season', 'Sport']:
    olympic_df[column] = label_encoder.fit_transform(olympic_df[column])

numerical_cols = olympic_df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = olympic_df[numerical_cols]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, ax=ax)
st.pyplot(fig)


############### MODEL ###############

st.header("Model")
st.write("Goal: Predicting if an athlete will win a medal or not using classification methods.")
st.subheader("")
st.markdown(""" **Prepare data**
            - Firstly I prepared the data for models by converting that ones into numerical values.
            - Defined target column: *Athlete Won Medal* 
            - Scaled the data to ensure that all features contribute equally to the model.
            """)

target_columns = ['Athlete Won Medal']

# Scale data
numerical_cols = [col for col in olympic_df.select_dtypes(include=['float64', 'int64']).columns if col not in target_columns]
scaler = MinMaxScaler()
olympic_df[numerical_cols] = scaler.fit_transform(olympic_df[numerical_cols])

# Model Building
st.subheader("Model Building")
st.markdown("""We have to drop the target column because we will ppredict them using models and other columns that are not useful for our analysis.
            """)
# X = features, y = target
X = olympic_df.drop(['ID', 'Name','Athlete Won Medal', 'Medal_Bronze', 'Medal_Gold', 'Medal_Silver', 'Medal_No medal', 'Team', 'NOC', 'reg', 'City', 'Event'], axis=1)
y = olympic_df['Athlete Won Medal']

# Model function
def make_prediction(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    actual_predicted_values = pd.DataFrame(dict(actual=y_test, prediction=predictions))
    # pd.crosstab(index=actual_predicted_values['actual'], columns=actual_predicted_values['prediction'])
    precision = metrics.precision_score(y_test, predictions)
    print(classification_report(y_test, predictions))
    

    return actual_predicted_values, accuracy, precision 

st.markdown("Using different classification methods")
st.markdown("1. Random Forest Classifier (with diffferent estimator nr and depth)")
model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
results_rf, accuracy_rf, precision_rf, class_report_rf = make_prediction(X, y, model)
st.write(f"Accuracy: {accuracy_rf:.2f}")
st.write(f"Precision: {precision_rf:.2f}")
st.write(class_report_rf)


model = RandomForestClassifier(n_estimators=250, min_samples_split=10, random_state=200)
make_prediction(X, y, model)

st.markdown("2. Logistic Regression")
model = LogisticRegression()
results_lr, accuracy_lr, precision_lr, class_report_lr = make_prediction(X, y, model)
st.write(f"**Accuracy:** {accuracy_lr:.2f}")
st.write(f"**Precision:** {precision_lr:.2f}")
st.table(class_report_lr)


st.markdown("3. XGBClassifier")
model=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
results_xgb, accuracy_xgb, precision_xgb, class_report_xgb = make_prediction(X, y, model)
st.write(f"Accuracy: {accuracy_xgb:.2f}")
st.write(f"Precision: {precision_xgb:.2f}")
st.dataframe(results_xgb)


############### CONCLUSION ###############
st.header("CONCLUSION")