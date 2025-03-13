import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le

titanic=pd.DataFrame(sns.load_dataset('titanic'))

mean_age = titanic['age'].mean()

missing_values = titanic.isnull().sum()

# Filling Missing Values

titanic.loc[:, 'age'] = titanic['age'].fillna(titanic['age'].median())
titanic.loc[:, 'embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
titanic.loc[:, 'deck'] = titanic['deck'].fillna(titanic['deck'].mode()[0])

# Drop Duplicates
titanic = titanic.drop_duplicates()

# Changes Age into type int
titanicage = titanic['age'].astype(int)
titanic.loc[:, 'age'] = titanicage

# Custom Function to detect outliers using IQR
def detect_outliers_iqr(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

# Defines and removes outliers identified with the above function
fare_lower_bound, fare_upper_bound = detect_outliers_iqr(titanic['fare'])
fare_outliers = titanic[(titanic['fare'] < fare_lower_bound) | (titanic['fare'] > fare_upper_bound)]
titanic.loc[:, 'fare'] = titanic['fare'].mask((titanic['fare'] < fare_lower_bound) | (titanic['fare'] > fare_upper_bound), titanic['fare'].median())

# Uses Label Encoder from sklearn to encode to numerical values 
label_encoder =le()
titanic.loc[:, 'embarked'] = label_encoder.fit_transform(titanic.loc[:, 'embarked']) + 1

# Creates family_size using sibsp and parch
titanic.loc[:, 'family_size'] = titanic.loc[:, 'sibsp'] + titanic.loc[:, 'parch'] + 1

# Creates age category using defined bins and labels
bins = [0, 12, 19, 60, 120]
labels = ['Child', 'Teen', 'Adult', 'Senior']
titanic.loc[:, 'age_category']  = pd.cut(titanic.loc[:, 'age'] , bins=bins, labels=labels, right=True)

titanic