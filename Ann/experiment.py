import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# load the dataset
data = pd.read_csv('Churn_Modelling.csv')
# print(data.head())
# drop unwanted columns
data=data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# encode categorical variable
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])


# one hot encoding for geography column
one_hot_encoder = OneHotEncoder()
data_geography = one_hot_encoder.fit_transform(data[['Geography']])
print(one_hot_encoder.get_feature_names_out(['Geography']))

geo_df = pd.DataFrame(data_geography.toarray(), columns=one_hot_encoder.get_feature_names_out(['Geography']))
data = pd.concat([data, geo_df], axis=1)
print(data.head())

with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)
    
with open('geo_encoder.pkl', 'wb') as f:
    pickle.dump(one_hot_encoder, f)

# divide data into dependent and independent features
x = data.drop(['Exited', 'Geography'], axis=1)
y = data['Exited']
# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)