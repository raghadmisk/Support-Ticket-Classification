# Support-Ticket-Classification

## Project Overview
This project implements a machine learning model to classify customer support tickets into predefined queues.

The system uses word processing and an SVM classifier to determine the appropriate category for new tickets.

 ## Objectives
- Cleaning and processing text ticket data
- Converting text to numerical properties using TF-IDF
- Training an SVM classification model
- Saving and reloading the trained model
- Predicting new incoming ticket categories

## Technologies Used
- Python
- Pandas
- Syct-Learn
- Joblip
- NumPay

## Model Details
- Text-to-vector conversion: TF-IDF (with n-grams)
- Classification algorithm: Supporting vector machine (SVM)
- Class balancing application
- Training/testing split: 80% training / 20% testing

## How to Run the Project

1. Ensure all files are in the same folder:

- main.py

- v.2.0.pkl

- vectorizer.pkl

- new_data.csv

2. Install the required libraries:
   pip install -r requirements.txt

3. Run the main file:
   python main.py

4. The predictions will be saved in the following file:
   tickets_with_predictions.csv

## Output
The system creates a CSV file containing the original tickets and the predicted support queue.
