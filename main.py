from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error
import numpy as np
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
dataset=pd.read_csv('/home/haroon/ameen/sarcasm project/final_dataset.csv')
y=dataset['target']
x=dataset.iloc[:,:-1]
x=x.drop('Unnamed: 0',axis=1)
x.info()
y.info()
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=42)
model=Sequential([
    Dense(128,activation='tanh',kernel_regularizer=regularizers.l1_l2(0.01),input_shape=(3,)),
    Dense(64,activation='tanh',kernel_regularizer=regularizers.l1_l2(0.01)),
    Dense(32,activation='tanh',kernel_regularizer=regularizers.l1_l2(0.01)),
    Dense(1)
])
model.compile(optimizer='adam',loss='mse',metrics=['mae','mse','mape'])
earlystopping=EarlyStopping(monitor='val_loss',patience=15)
reducelronpleataue=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=15,min_lr=0.000001)
model.fit(x_train,y_train,validation_split=0.3,epochs=100,batch_size=32)
loss,mae,mse,mape=model.evaluate(x_test,y_test)
print("loss",loss)
print("mae",mae)
print("mse",mse)
print("mape",mape)
y_pred=model.predict(x_test)
r2=r2_score(y_test,y_pred)
print("r2",r2)
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, y_pred):.4f}")