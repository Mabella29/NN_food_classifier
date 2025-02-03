# Food Image Classification with VGG19

## Problem Description
This project aims to classify food images into 10 categories using a pre-trained VGG19 model. The dataset used is a subset of the `10_food_classes` dataset, containing 10% of the original data. The goal is to build a model that can accurately classify food images while avoiding overfitting.

## Dataset
The dataset can be downloaded from [this link](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip). Extract the dataset into the `data/` folder. The folder structure should look like this:


## How to Use the Files

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

### Step-by-Step Instructions

1. **Set Up the Environment**:
   - Clone this repository:
     ```bash
     git clone https://github.com/Mabella29/NN_food_classification.git
     cd food_classification_project
     ```
   - Create and activate a virtual environment (optional but recommended):
     ```bash
     # On Windows
     python -m venv venv
     venv\Scripts\activate

     # On macOS/Linux
     python -m venv venv
     source venv/bin/activate
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Explore the Dataset and Preprocess the Data**:
   - Open the Jupyter Notebook to explore the dataset and perform data preparation:
     ```bash
     jupyter notebook notebook.ipynb
     ```
   - Run all cells in the notebook to preprocess the data and perform exploratory data analysis (EDA).

3. **Train the Model**:
   - Run the `train.py` script to train the VGG19 model. The trained model will be saved to `models/vgg19_food_classifier.h5`:
     ```bash
     python train.py
     ```

4. **Make Predictions**:
   - Run the `predict.py` script to load the trained model and make predictions on new images:
     ```bash
     python predict.py
     ```

5. **Start the Web Service** (Optional):
   - Run the Flask app to serve the model as a web service:
     ```bash
     python app/app.py
     ```
   

6. **Stop the Web Service**:
   - Press `Ctrl+C` in the terminal where the Flask app is running to stop the web service.

## Dependencies
All dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt