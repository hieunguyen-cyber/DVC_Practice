import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(data_url, names=columns)
df.to_csv("./data/raw.csv", index=False)

scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

df.to_csv('./data/processed.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("./data/train.csv", index=False)
test_df.to_csv("./data/test.csv", index=False)