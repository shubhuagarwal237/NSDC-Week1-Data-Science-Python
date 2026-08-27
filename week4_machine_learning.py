import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('titanic_cleaned.csv')
df['sex']=LabelEncoder().fit_transform(df['sex'].astype(str))
X=df[['pclass','sex','age','fare']].fillna(df[['pclass','sex','age','fare']].median())
y=df['survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
