![image](https://github.com/user-attachments/assets/4639c146-9958-4655-9f0f-d7fcd8fda5b2)

# NBA Performance Analysis (2012-2024) : Insights into Player and Team Evolution
## Project Objectives
* Identifying trends and patterns in player and team performance over the last decade.
* Analyzing the impact of key stats (e.g., 3-point shooting, assists, rebounds) on winning percentages in both regular season and playoffs.
* Evaluating player efficiency and identify the most valuable players (MVPs) by season, both quantitatively (data-driven) and qualitatively.
* Building predictive insights to suggest areas where teams or players can improve to increase championship chances.
* Creating an interactive dashboard to showcase findings dynamically for stakeholder decision-making.

## Feature Engineering
- Creating working environment (nba_envi) in the folder so that all the required datasets and dependencies remain in the same folder.
- Downloading kaggle library (pip install kaggle)
- Downloading dataset from kaggle using API link ( kaggle datasets download -d {paste the link} )
- Downloading dependencies
```python
# Importing Dependencies
import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sb
```
##  Data Exploration and Cleaning
**Objective:** Ensure the dataset is clean, consistent, and ready for analysis.
**Steps:**
1. Checking for missing or inconsistent values (e.g., FG_PCT or FT_PCT) and handle them (e.g., imputation or dropping rows if necessary).
2. Converting data types where required (e.g., YEAR as integer, Season_type as a categorical variable).
3. Adding derived columns, such as:
* Points Per Game (PPG) = PTS / GP
* Assist-to-Turnover Ratio (AST_TOV_Ratio) = AST / TOV
* Rebound Efficiency (REB_EFF) = (OREB + DREB) / Total Rebounds Available
* True Shooting Percentage (TS%):
  Formula: TS% = PTS/2x(FGA+0.44XFTA)

```python
#Importing and Reading data
df = pd.read_csv('nba.csv', encoding_errors='ignore')
af = pd.read_csv('Playoffs.csv', encoding_errors='ignore')
bf = pd.read_csv('Regular_Season.csv', encoding_errors='ignore')

# Data Exploration & Cleaning
df.info()
df.describe()

# Checking for NULL values
df.isnull().sum()

# Checking for Duplicate Values
df.duplicated().sum()

# Correcting the values of data 
df['Season_type']= df['Season_type'].str.replace('%20', ' ')
df['Season_type']


# Checking & Changing data types of columns
df.columns
df.dtypes

# Changing Datatypes of required coloumns
# Spliting YEAR into START_YEAR and END_YEAR
df[['START_YEAR', 'END_YEAR']] = df['year'].str.split('-', expand=True)
df['START_YEAR'] = df['START_YEAR'].astype(int)
df['END_YEAR'] = df['END_YEAR'].astype(int)

df['Season_type']=df['Season_type'].astype('category')

# Adding derived coloums. 1. PPG(Points per game) = PTS(Points)/GP(Games Played)
df['PPG'] = df['PTS']/df['GP']
# Handling division by zero (if GP is 0, set PPG to NaN or 0)
df['PPG'] = df['PPG'].fillna(0) 
# Assist-to-Turnover Ratio (AST_TOV_Ratio) = AST / TOV
df['AST_TOV_Ration'] = df['AST']/df['TOV']

# Rebound Efficiency (REB_EFF) = (OREB + DREB) / Total Rebounds Available

df['Missed_FGs'] = df['FGA'] - df['FGM']  # Missed Field Goals
df['Missed_FTs'] = df['FTA'] - df['FTM']  # Missed Free Throws

# Approximation for Total Rebounds Available
df['Total_Rebounds_Available'] = df['Missed_FGs'] + df['Missed_FTs']

# Rebound Efficiency (REB_EFF)
df['REB_EFF'] = (df['OREB'] + df['DREB']) / df['Total_Rebounds_Available']
```
## 2. Exploratory Data Analysis (EDA)
**Objective:** Understanding patterns and trends in the data.

**Analysis Ideas:**
**Performance Trends:**
* How have key metrics (e.g., FG%, 3P%, REB, AST) changed from 2012 to 2024?
* Comparing the evolution of player performance in regular seasons vs. playoffs.
  
**Team Analysis:**
* Identifying the top-performing teams based on cumulative stats like total points, rebounds, and assists.
* Evaluating how team strategies (e.g., reliance on 3-point shooting) evolved over time.
  
**Player Insights:**
* Ranking players by PPG, REB, AST, and other key stats per year.
* Correlate individual stats with their team’s winning percentage.
* Highlighting overachievers (players with high Efficiency Rating (EFF) in fewer minutes played).

**Season Comparison:**
* Comparing average stats between regular season and playoffs.
* Identifying players who consistently outperform during playoffs.
Tools: Python (Pandas, Matplotlib, Seaborn), SQL.
Deliverables:
EDA visuals (e.g., line plots for trends, heatmaps for correlations, bar charts for top players/teams).
