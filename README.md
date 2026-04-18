# Forest Fire Weather Index Prediction

This project is a Flask web app that predicts the Forest Fire Weather Index
using a trained Ridge Regression model and a saved StandardScaler.

## Project Structure

```text
.
|-- application.py
|-- requirements.txt
|-- Models/
|   |-- ridge.pickle
|   `-- scaler.pickle
|-- templates/
|   |-- index.html
|   `-- home.html
`-- notebook/
    |-- Algerian_forest_clean_data.csv
    `-- model_training.ipynb
```

## Requirements

- Python 3.10 or newer
- Flask
- numpy
- pandas
- scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Flask app with:

```bash
python application.py
```

Then open:

```text
http://127.0.0.1:5000/predictdata
```

## Input Features

The prediction form uses these values:

- Temperature
- RH
- Ws
- Rain
- FFMC
- DMC
- ISI
- Classes
- Region

The app scales the input data with `Models/scaler.pickle` and predicts the Fire
Weather Index with `Models/ridge.pickle`.
