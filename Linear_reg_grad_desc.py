
import pylab
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd 


data = pd.read_csv('ex1data1.txt',header=None)

data = data.rename(columns = {0:'Profit',1:'Population'})

print (data.head())

Y_train,Y_test,X_train,X_test, = train_test_split(data["Profit"],data["Population"],test_size=0.2,random_state=10)

print(X_train.shape)
print(X_test.shape)



data['z'] = 1

test_arr = data['z'][0:77]
test_arr2 = data['z'][0:20]

print(data.head())


# Linear Regression with Gradient Descent
No_of_Iterations = 1000

b0 = 0 
b1 = 0

Learning_Rate = 0.0005

for i in range(No_of_Iterations):
	cost = 0
	for x_i,y_i in zip(X_train,Y_train):
		error = b0+(b1*x_i) - y_i
		b0 = b0 - Learning_Rate*error
		b1 = b1 - Learning_Rate*error*x_i
		cost += error**2
	if(i%10==0):
		print('No_Iter=%d, cost = %0.3f , b0 = %0.3f , b1=%0.3f'%(i,cost,b0,b1))


Y_pred = X_test*b1+b0

plt.scatter(X_test,Y_test,color='black')
plt.plot(X_test,Y_pred,color='blue',linewidth=1)

plt.xticks(())
plt.yticks(())
pylab.show()
newarr=np.vstack((X_train,test_arr)).T
newarr2 = np.vstack((X_test,test_arr2)).T
#print(newarr)

#Testing against sklearn linear regression

regr = linear_model.LinearRegression()

print(X_train.shape)

regr.fit(newarr,Y_train)

y_pred = regr.predict(newarr2)

plt.scatter(X_test,Y_test,color='black')
plt.plot(X_test,y_pred,color='blue',linewidth=1)

plt.xticks(())
plt.yticks(())
pylab.show()

print('Coefficients: \n', regr.coef_)
