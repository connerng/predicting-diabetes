import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('diabetes_data.csv')
y = df['Diagnosis']
x = df.drop('Diagnosis', axis=1)
x = x.drop('DoctorInCharge', axis=1)
x = x.drop('PatientID', axis=1)
x = x.drop('Ethnicity', axis=1)
x = x.drop('SocioeconomicStatus', axis=1)
x = x.drop('EducationLevel', axis=1)
x = x.drop('HealthLiteracy', axis=1)
x = x.drop('MedicalCheckupsFrequency', axis=1)

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=61)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print('Training MSE: ', train_mse)
print('Training R^2: ', train_r2)
print('Test MSE: ', test_mse)
print('Test R^2: ', test_r2)

lr_results = pd.DataFrame(['Linear Regression', train_mse, train_r2, test_mse, test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R^2', 'Test MSE', 'Test R^2']

print(lr_results)


