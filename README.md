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

**Performance Trends:**
1. How have key metrics (e.g., FG%, 3P%, REB, AST) changed from 2012 to 2024?
To analyze how key metrics (e.g., FG%, 3P%, REB, AST) have changed over the years from 2012 to 2024, we will:

* Group Data by Year: Aggregate the metrics by the YEAR column.
* Calculate Mean or Median: Compute the yearly average or median for each key metric to observe trends.
* Plot Trends: Visualize the trends over time for better understanding.
```python
# Creating new coloumns for FG% and 3P%
nf['FG%'] = (nf['FGM'] / nf['FGA']) * 100
nf['3P%'] = (nf['FG3M'] / nf['FG3A']) * 100

#Ensuring start year should be numeric
nf['START_YEAR'] = pd.to_numeric(nf['START_YEAR'], errors='coerce')

#Grouping by start year and calulcating mean
metrics = ['FG%', '3P%', 'REB', 'AST']
yearly_trends = nf.groupby('START_YEAR')[metrics].mean().reset_index()

#Plotting the trend
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for metric in metrics:
    plt.plot(yearly_trends['START_YEAR'], yearly_trends[metric], label=metric)

# Adding titles and labels
plt.title("Yearly Trends of Key Metrics (2012-2024)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Value", fontsize=12)
plt.legend(title="Metrics", fontsize=10)
plt.grid(alpha=0.3)
plt.show()
```
![Yearly trends of key metrics (2012-2024)](https://github.com/user-attachments/assets/eff13562-fc8b-4a6f-854f-01cbbc25fdc9)

2.  Compare the evolution of player performance in regular seasons vs. playoffs
* Calculating the yearly trends for key metrics (e.g., PTS, AST, REB, FG%)
* Calculating Separately for Regular Season and Playoffs
```python
# Defining key metrics to compare
metrics1 = ['PTS', 'AST', 'REB', 'FG%']

# Grouping data by START_YEAR and Season_type, then calculating mean for each metric
performance_trends = (nf.groupby(['START_YEAR', 'Season_type'])[metrics1]
                      .mean()
                      .reset_index()
                    )

# Separate data for Regular Season and Playoffs
regular_season = performance_trends[performance_trends['Season_type'] == 'RegularSeason']
playoffs = performance_trends[performance_trends['Season_type'] == 'Playoffs']

# Plotting the trends
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))

# Plot each metric for Regular Season and Playoffs
for metric in metrics1:
    plt.plot(regular_season['START_YEAR'], regular_season[metric], label=f"Regular Season - {metric}", linestyle='--', marker='o')
    plt.plot(playoffs['START_YEAR'], playoffs[metric], label=f"Playoffs - {metric}", linestyle='-', marker='x')

# Add labels, legend, and grid
plt.title("Evolution of Player Performance: Regular Season vs. Playoffs (2012-2024)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Metric Value", fontsize=12)
plt.legend(title="Metrics and Season Type", fontsize=10)
plt.grid(alpha=0.3)
plt.show()
```
![image](https://github.com/user-attachments/assets/de7cac8b-35c4-41b8-a812-be0f0af6b6a1)

**Team Analysis:**
1. Identify the top-performing teams based on cumulative stats like total points, rebounds, and assists.
```python
# Defining key metrics to compare stats
team_metrics = ['PTS', 'REB', 'AST' ]
# Grouping by team
team_performance = (nf.groupby(['TEAM'])[team_metrics]
                    .mean()
                    .reset_index()
                    .sort_values(by='PTS', ascending=True)
                )

# Adding a rank column based on total points
team_performance['Rank'] = team_performance['PTS'].rank(ascending=True, method='dense').astype(int)

# Display top-performing teams
top_teams = team_performance.head(10)  # Adjust the number as needed for top-N teams
print(top_teams)

# Plot the performance of top teams
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
x = top_teams['TEAM']
plt.bar(x, top_teams['PTS'], color='skyblue', label='Total Points')
plt.bar(x, top_teams['REB'], color='orange', label='Total Rebounds', bottom=top_teams['PTS'])
plt.bar(x, top_teams['AST'], color='green', label='Total Assists', bottom=top_teams['PTS'] + top_teams['REB'])
plt.xlabel('Team Name')
plt.ylabel('Cumulative Stats')
plt.title('Top Performing Teams Based on Cumulative Stats')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/f4072dd8-0b2b-4007-a4c8-472d6ef51a86)
2. Evaluate how team strategies (e.g., reliance on 3-point shooting) evolved over time. 
To evaluate how team strategies, such as reliance on 3-point shooting, evolved over time, we can analyze metrics like 3-point attempts (3PA), 3-point made (3PM), and 3-point percentage (3P%). 
We will compare these metrics across years and visualize their trends.
```python
import matplotlib.pyplot as plt

# Relevant metrics
metrics3 = ['FG3A', 'FG3M', 'PTS']

# Grouping by START_YEAR to analyze trends over time
team_strategy = (
    nf.groupby('START_YEAR')[metrics3]
    .sum()
    .reset_index()
)

# Adding derived metrics
team_strategy['3P_Reliance'] = team_strategy['FG3M'] / team_strategy['PTS']

# Plot trends for 3PA, 3PM, and 3P_Reliance
plt.figure(figsize=(14, 8))
#plt.plot(team_strategy['START_YEAR'], team_strategy['FG3A'], label='3-Point Attempts', marker='o')
#plt.plot(team_strategy['START_YEAR'], team_strategy['FG3M'], label='3-Point Made', marker='o')
plt.plot(team_strategy['START_YEAR'], team_strategy['3P_Reliance'], label='3P Reliance (%)', marker='o')

# Formatting the plot
plt.title('Evolution of Team Strategies (3-Point Shooting) Over Time', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/904239e0-01ac-43b6-af1c-24b4b321fe3e)

```python
import matplotlib.pyplot as plt

# Relevant metrics
metrics4 = ['FG3A', 'FG3M', 'PTS']

# Grouping by TEAM and START_YEAR
team_strategy1 = (
    nf.groupby(['TEAM', 'START_YEAR'])[metrics4]
    .sum()
    .reset_index()
)

# Adding derived metrics
team_strategy1['3P_Reliance'] = team_strategy1['FG3M'] / team_strategy1['PTS']

# Filter data for top teams (e.g., based on total points or specific teams)
top_teams = ['GSW', 'CLE', 'LAL', 'BOS']  # Replace with desired teams
filtered_data = team_strategy1[team_strategy1['TEAM'].isin(top_teams)]

# Plot trends for top teams
plt.figure(figsize=(14, 8))
for team in top_teams:
    team_data = filtered_data[filtered_data['TEAM'] == team]
    plt.plot(team_data['START_YEAR'], team_data['3P_Reliance'], label=f'{team} 3P Reliance', marker='o')

# Formatting the plot
plt.title('Evolution of Team Strategies (3-Point Reliance) Over Time', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('3-Point Reliance (%)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/2f4094a1-d1cc-4b5e-81de-24ffdc3c6675)
3. rank no. 1 from each year and his ppg in graph
To visualize the top-ranked player's PPG for each season, we can extract the player with rank 1 for each year and plot their Points Per Game (PPG) over time.
```python
import matplotlib.pyplot as plt

nf['PPG_Rank'] = nf.groupby('START_YEAR')['PPG'].rank(ascending=False, method='dense').astype(int)

top_players = nf[nf['PPG_Rank'] == 1][['START_YEAR', 'PLAYER', 'PPG']]

# Plotting the trend
plt.figure(figsize=(14, 8))
plt.plot(top_players['START_YEAR'], top_players['PPG'], marker='o', label="Top Player's PPG")

# Adding labels to the points
for i, row in top_players.iterrows():
    plt.text(row['START_YEAR'], row['PPG'], row['PLAYER'], fontsize=8, ha='right')
plt.title("Top Player's PPG by Season", fontsize=16)
plt.xlabel("Season (Year)", fontsize=12)
plt.ylabel("Points Per Game (PPG)", fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/99aa7219-8abe-418d-9161-1d51dd7fcd7f)
 4.  
Steps to Highlight Overachievers
Compute Efficiency Rating 
Useing a formula like:
EFF=(PTS+REB+AST+STL+BLK−(FGA−FGM)−(FTA−FTM)−TO)/MIN

Filtering Players:
Focusing on players with fewer minutes played, e.g., MIN below a certain threshold (e.g., less than the median).

Visualizing:
Ploting a scatter plot of MIN vs. EFF.
Highlight players with high EFF but low MIN.

Rank Overachievers:
Sort by EFF in descending order for players with fewer minutes.

 ```python
# Calculating Efficiency Rating (EFF)
nf['EFF'] = (
    nf['PTS'] + nf['REB'] + nf['AST'] + nf['STL'] + nf['BLK'] -
    (nf['FGA'] - nf['FGM']) - (nf['FTA'] - nf['FTM']) - nf['TOV']
) / nf['MIN']

# Filtering players with fewer minutes (e.g., below median minutes)
low_minutes_players = nf[nf['MIN'] < nf['MIN'].median()]

# Sorting these players by EFF
overachievers = low_minutes_players.sort_values(by='EFF', ascending=False)

# Displaying the top 10 overachievers
print("Top Overachievers (High EFF in Low Minutes):")
print(overachievers[['PLAYER', 'MIN', 'EFF']].head(10))

# Visualizing Overachievers
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(low_minutes_players['MIN'], low_minutes_players['EFF'], alpha=0.6, color='skyblue', edgecolor='black')
plt.title('Overachievers: High EFF in Fewer Minutes')
plt.xlabel('Minutes Played')
plt.ylabel('Efficiency Rating (EFF)')
plt.axhline(y=low_minutes_players['EFF'].median(), color='red', linestyle='--', label='Median EFF')
plt.axvline(x=low_minutes_players['MIN'].median(), color='green', linestyle='--', label='Median MIN')
plt.legend()
plt.grid(alpha=0.7)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/568a3404-67d2-4bc6-9bdf-e8eeaf2c2301)

![image](https://github.com/user-attachments/assets/6178382d-b813-46bf-a8da-4aca7edcc93f)
5. Season 
