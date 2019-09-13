# Disaster Response Pipelines

## Project Description
During disaster events, many disaster response organizations try to understand the messages sent by people and help accordingly. However, due to the extensiveness of these mesasges, it will be useful to have a technology that can classify these messages, so these organizations can help in a more efficient way. In this project, I focused on analyzing disaster data from Figure Eight and classify messages sent during disaster events usiung machine learning techniques. 

I created ETL pipeline that loads and cleans datasets, and makes it ready for training a model. Later, the text data was trained and tested using machine learning - mutlioutput classifier. A web app was developed that displays model results as charts, and also allows users to classify disaster messages.

## Files
Main files that are used in this project are
* Messages and Categories CSV files (datasets)
* process_data.py: This is ETL pipleine python script that cleans your datasets and loads them to a SQL database. Jupyter notebook file for this is ETL Pipeline Preparation.ipynb
* train_classifier.py: This is machine learning pipeline python script that actually trains and tests your models ability to classify your text messages as accurately as possible. Jupyter notebook file for this is ML Pipeline Preparation.ipynb
* Finally the files for webapp- run.py python script, which allows your model results to be displayed as visual summaries on a webapp, where users can even type their own disaster messages to classify them

## Instructions to run the files:
1) Run the following commands in the project's root directory to set up your database and model.

  To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv    data/disaster_categories.csv data/disaster_message_categories.db
  To run ML pipeline that trains classifier and saves python models/train_classifier.py data/disaster_message_categories.db models/model.p
2) Run the following command in the app's directory to run your web app. python run.py

3) Go to http://0.0.0.0:3001/

## Results from webapp
![Test Image 1](https://github.com/Jyothsnajo/Disaster-Response-Pipelines-DSND/blob/master/Images/MessageClassifier.PNG)
![Test Image 2](https://github.com/Jyothsnajo/Disaster-Response-Pipelines-DSND/blob/master/Images/Charts.PNG)

