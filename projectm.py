import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import io

# Load the dataset
file_path = r"C:\Users\Y SAI KUMAR\Downloads\mtcars (1).xlsx"
df = pd.read_excel(file_path)

# Calculate min and max values for each relevant field
min_max_values = {
    'cyl': (df['cyl'].min(), df['cyl'].max()),
    'disp': (df['disp'].min(), df['disp'].max()),
    'hp': (df['hp'].min(), df['hp'].max()),
    'drat': (df['drat'].min(), df['drat'].max()),
    'wt': (df['wt'].min(), df['wt'].max()),
    'qsec': (df['qsec'].min(), df['qsec'].max()),
    'gear': (df['gear'].min(), df['gear'].max()),
    'carb': (df['carb'].min(), df['carb'].max())
}

# Function to create an image with a number
def create_number_image(number, width_cm, height_cm):
    # Convert cm to inches for Matplotlib
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.text(0.5, 0.5, str(number), fontsize=42, ha='center', va='center')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return buf

# Streamlit app
st.image(r"C:\Users\Y SAI KUMAR\Music\innomatics-footer-logo.webp")
st.title("Car Mileage Prediction")

# Load the model
model = pickle.load(open(r"C:\Users\Y SAI KUMAR\New folder\lin.pkl", "rb"))

# Input fields for car details with min and max values
cyl = st.number_input("Enter the number of cylinders", min_value=min_max_values['cyl'][0], max_value=min_max_values['cyl'][1])
disp = st.number_input("Enter the size of the Engine", min_value=min_max_values['disp'][0], max_value=min_max_values['disp'][1])
hp = st.number_input("Enter the measure of the power produced by the Engine", min_value=min_max_values['hp'][0], max_value=min_max_values['hp'][1])
drat = st.number_input("Enter the rear axle ratio of the cars", min_value=min_max_values['drat'][0], max_value=min_max_values['drat'][1])
wt = st.number_input("Enter the weight of the car", min_value=min_max_values['wt'][0], max_value=min_max_values['wt'][1])
qsec = st.number_input("Enter the measure of quarter-mile time of the car", min_value=min_max_values['qsec'][0], max_value=min_max_values['qsec'][1])

vs = st.radio("The shape of the Engine", ['V-shaped Engine', 'Straight-line Engine'])
shape = 0 if vs == "V-shaped Engine" else 1

am = st.radio("Type of transmission of the car", ['Automatic Transmission', 'Manual Transmission'])
type = 0 if am == 'Automatic Transmission' else 1 

gear = st.number_input("Enter the number of Gears of the car", min_value=min_max_values['gear'][0], max_value=min_max_values['gear'][1])
carb = st.number_input("Enter the number of Carburetors of the car", min_value=min_max_values['carb'][0], max_value=min_max_values['carb'][1])

if st.button("Predict Mileage"):
    # Predict the mileage
    mpg = model.predict([[cyl, disp, hp, drat, wt, qsec, shape, type, gear, carb]])[0]

    st.write("The Mileage for the car with the given details is:", mpg)

    # Generate and display the image with specified size
    image_buf = create_number_image(mpg, width_cm=5, height_cm=2)
    st.image(image_buf, use_column_width=True)
