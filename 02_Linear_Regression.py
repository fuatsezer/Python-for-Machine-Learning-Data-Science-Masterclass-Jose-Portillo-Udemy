import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error



df = pd.read_csv(".\datasets\Advertising.csv")
print(df.head())

df["total_spend"] = df["TV"] + df["newspaper"] + df["radio"]
print(df.head())

sns.regplot(data=df, x= "total_spend",y= "sales");

X = df["total_spend"]
y = df["sales"]
# y = mx + b
# y = B1x + B0
# help(np.polyfit)

np.polyfit(X,y,deg=1)
potential_spend = np.linspace(0,500,100)

predicted_sales = np.polyfit(X,y,deg=1)[0] * potential_spend + np.polyfit(X,y,deg=1)[1]

sns.scatterplot(x="total_spend",y = "sales", data= df)
plt.plot(potential_spend,predicted_sales,color="red");


fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].set_title("Newspaper Spend");
axes[2].set_ylabel("Sales")
plt.tight_layout();

# Relationships between features
sns.pairplot(df,diag_kind='kde');

X = df.drop("sales",axis=1)
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)
test_predictions = model.predict(X_test)

df["sales"].mean()

sns.histplot(data=df,x="sales");

mean_absolute_error(y_test,test_predictions)

np.sqrt(mean_squared_error(y_test,test_predictions))

test_residuals = y_test - test_predictions

sns.scatterplot(x=y_test,y=test_residuals)
plt.axhline(y=0,color="red",ls="--");

sns.displot(test_residuals,bins=25,kde=True);