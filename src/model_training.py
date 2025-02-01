# importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

#Load and preprocess the data
features_train, features_test, target_train, target_test = generate_and_process_data()

#initialize the ANN
classifier=Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer="he_uniform",activation="relu",input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer="he_uniform",activation="relu"))

# Adding the outpyut layer
classifier.add(Dense(units=1,kernel_initializer="glorot_uniform",activation="sigmoid"))

#Compiling the ANN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Fitting the ANN to the training dataset
model_history=classifier.fit(features_train,target_train,validation_split=0.33,batch_size=10,epochs=20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
target_pred = classifier.predict(features_test)
target_pred = (target_pred > 0.5)

# Save the model
classifier.save('./model/classification_model.h5')

# Evaluate the model and write the results to a file
loss, accuracy = classifier.evaluate(features_test, target_test)
with open('./reports/model_metrics.txt', 'w') as f:
    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")

print("Model training completed and metrics saved.")


