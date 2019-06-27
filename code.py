# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Code starts here
df = pd.read_csv(path)

print(df.columns)
df = df[['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Preferred Positions', 'Wage']]

# Code ends here


# --------------
# Removes the symbol from values
df['Unit'] = df['Value'].str[-1]
df['Value (M)'] = np.where(df['Unit'] == '0', 0, df['Value'].str[1:-1].replace(r'[a-zA-Z]', ''))
df['Value (M)'] = df['Value (M)'].astype(float)
df['Value (M)'] = np.where(df['Unit'] == 'M', df['Value (M)'], df['Value (M)']/1000)

# Removes the symbol from Wage
df['Unit2'] = df['Wage'].str[-1]
df['Wage (M)'] = np.where(df['Unit2'] == '0', 0, df['Wage'].str[1:-1].replace(r'[a-zA-Z]', ''))
df['Wage (M)'] = df['Wage (M)'].astype(float)
df['Wage (M)'] = np.where(df['Unit2'] == 'M', df['Wage (M)'], df['Wage (M)']/1000)

# Drop the Unit and Unit2 from df
df = df.drop(['Unit', 'Unit2'], 1)

# New column position
df['Position'] = df['Preferred Positions'].str.split().str[0]


# --------------
import seaborn as sns

print(df.columns)
df_group = df.groupby(['Position']).sum()
# Code starts here
sns.countplot(x='Position', data=df)

value_distribution_values = df.sort_values("Wage (M)", ascending=False).reset_index().head(100)[["Name", "Wage (M)"]]
sns.countplot(x='Wage (M)', data=value_distribution_values)
# value_distribution_values = df[]

overall = df.sort_values("Overall")

overall_value = overall.groupby(['Overall'])['Value (M)'].mean()
# Code ends here


# --------------

p_list_1= ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

p_list_2 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']



    
# p_list_1 stats
df_copy = df.copy()
store = []
for i in p_list_1:
    store.append([i,
                    df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(
                        index=False), df_copy[df_copy['Position'] == i]['Overall'].max()])
df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace=True)
# return store
df1= pd.DataFrame(np.array(store).reshape(11, 3), columns=['Position', 'Player', 'Overall'])


# p_list_2 stats
df_copy = df.copy()
store = []
for i in p_list_2:
    store.append([i,
                    df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(
                        index=False), df_copy[df_copy['Position'] == i]['Overall'].max()])
df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace=True)
# return store
df2= pd.DataFrame(np.array(store).reshape(11, 3), columns=['Position', 'Player', 'Overall'])

if df1['Overall'].mean() > df2['Overall'].mean():
        print(df1)
        print(p_list_1)
else:
    print(df2)
    print(p_list_2)
        
    
    
    


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split


# Code starts here
X = df[['Overall','Potential','Wage (M)']]
y = df['Value (M)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test,y_pred)
print("r2", r2)

mae = mean_absolute_error(y_test, y_pred)
print("mae", mae)

# Code ends here


# --------------
from sklearn.preprocessing import PolynomialFeatures

# Code starts here
poly = PolynomialFeatures(3)
X_train_2 = poly.fit_transform(X_train)
X_test_2 = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_2, y_train)

y_pred_2 = model.predict(X_test_2)

r2 = r2_score(y_test,y_pred_2)
print("r2", r2)

mae = mean_absolute_error(y_test, y_pred_2)
print("mae", mae)
# Code ends here


