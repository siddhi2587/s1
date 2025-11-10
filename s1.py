from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Predict & evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



























import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import accuracy_score

df=pd.read_csv(r"C:\Users\admin\Downloads\Mental Health Dataset.csv")
target_col='Country'
X=pd.get_dummies(df.drop(target_col,axis=1),drop_first=True)
y=df[target_col]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rf=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print(f"accuracy:{accuracy_score(y_test,y_pred)*100:2f}%")




































import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\admin\Downloads\Mental Health Dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(df['Country'], df['Gender'], test_size=0.2, random_state=42)

vec = CountVectorizer()
X_train_vec, X_test_vec = vec.fit_transform(X_train), vec.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_vec, y_train)
y_pred = mnb.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))















































import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r"C:\Users\admin\Downloads\Mental Health Dataset.csv")
X = pd.get_dummies(df.drop('Country', axis=1), drop_first=True)
y = df['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))
