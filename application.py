import pickle
from pathlib import Path
from flask import (
    Flask, 
    request, 
    render_template
)

application = Flask(__name__)

app = application

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'Models' / 'ridge.pickle', 'rb') as model_file:
    ridge_model = pickle.load(model_file)

with open(BASE_DIR / 'Models' / 'scaler.pickle', 'rb') as scaler_file:
    standard_scaler = pickle.load(scaler_file)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        input_data = pd.DataFrame(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
            columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC',
                     'ISI', 'Classes', 'Region']
        )
        new_scaled_data = standard_scaler.transform(input_data)

        result = ridge_model.predict(new_scaled_data)

        return render_template('home.html', results=round(result[0], 2))
       
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(
        host = '0.0.0.0',
        debug=True
    )
