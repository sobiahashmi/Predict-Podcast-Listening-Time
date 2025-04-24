import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Import Dataset
df_train = pd.read_csv("train.csv")
df_train.head()
print("Shape of the train data: ", df_train.shape)
df_test = pd.read_csv("test.csv")
df_test.head()
print("Shape of the test data: ", df_test.shape)

# Preprocess Data

## Check for missing values
df_train['Episode_Length_minutes'].fillna(df_train['Episode_Length_minutes'].mean(),inplace=True)
df_train['Guest_Popularity_percentage'].fillna(df_train['Guest_Popularity_percentage'].mean(),inplace=True)
df_train['Number_of_Ads'].fillna(df_train['Number_of_Ads'].mode()[0],inplace = True)

df_test['Episode_Length_minutes'].fillna(df_test['Episode_Length_minutes'].mean(),inplace=True)
df_test['Guest_Popularity_percentage'].fillna(df_test['Guest_Popularity_percentage'].mean(),inplace=True)

## Drop columns that are not needed
df_train.drop(['Episode_Title'], axis = 1, inplace = True)
df_test.drop(['Episode_Title'], axis = 1 , inplace = True)

df_train= df_train.drop(['Podcast_Name'],axis=1)
df_test = df_test.drop(['Podcast_Name'], axis=1)

df_train= df_train.drop(['id'],axis=1)
df_test = df_test.drop(['id'], axis=1)

# Label Encoding (Label/Target)
le = LabelEncoder()
# Encode features
df_train['Genre'] = df_train['Genre'].astype('category').cat.codes
df_train['Number_of_Ads'] = df_train['Number_of_Ads'].astype('category').cat.codes
df_train['Publication_Day'] = df_train['Publication_Day'].astype('category').cat.codes
df_train['Publication_Time'] = df_train['Publication_Time'].astype('category').cat.codes
df_train['Episode_Sentiment'] = df_train['Episode_Sentiment'].astype('category').cat.codes

# ## Convert categorical variables to numerical 'One Hot Encoding'
# df_train_clean = pd.get_dummies(df_train,columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment'])
# df_test = pd.get_dummies(df_test, columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment'])

# Select features and target variable ,'Host_Popularity_percentage', 'Guest_Popularity_percentage', 
                    # 'Podcast_Length', 'Episode_Length_minutes'
selected_features = ['Genre','Number_of_Ads','Publication_Day','Publication_Time','Episode_Sentiment']
X = df_train[selected_features]
y = df_train['Listening_Time_minutes']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X,y)

# Save the model
joblib.dump(model,'listening_model.pkl')
# Load the model
load_model = joblib.load('listening_model.pkl')

