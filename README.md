# Heart Stroke Prediction Using Artificial Neural Network (ANN) :

This project aims to develop a predictive model for heart stroke risk assessment using machine learning techniques. The provided dataset contains various features related to individuals' demographics, lifestyle, and medical history. By analyzing these features, the model predicts the likelihood of an individual experiencing a heart stroke. While various machine learning algorithms could be employed for this task, this project specifically focuses on utilizing Artificial Neural Networks (ANN) due to their ability to capture complex patterns and relationships within the data.

## **Dataset**

The heart stroke prediction dataset consists of the following columns:

- **Age**: The age of the individual.
- **Gender**: The gender of the individual (male or female).
- **Hypertension**: Whether the individual has hypertension (0 for no, 1 for yes).
- **Heart Disease**: Whether the individual has a history of heart disease (0 for no, 1 for yes).
- **Marital Status**: The marital status of the individual.
- **Work Type**: The type of work the individual is engaged in (e.g., private, self-employed, government job).
- **Residence Type**: Whether the individual lives in an urban or rural area.
- **Average Glucose Level**: The average glucose level in the individual's blood.
- **BMI**: Body Mass Index, a measure of body fat based on height and weight.
- **Smoking Status**: The smoking status of the individual (e.g., never smoked, formerly smoked, currently smoking).
- **Stroke**: The target variable indicating whether a stroke occurred (0 for no, 1 for yes).

## **Usage**

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/heart-stroke-prediction.git
```
**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```
**3. Run the main.py script:**
```bash
python main.py
```
## Results
After training the model using Artificial Neural Networks (ANN) on the provided dataset, it achieves an accuracy of 97.47% in predicting heart stroke occurrences. The model's performance is evaluated using various evaluation metrics such as precision, recall, and F1-score, and the model is better at it's performance.
