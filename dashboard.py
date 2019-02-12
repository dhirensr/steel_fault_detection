import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import steel_data_analysis
import datetime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Label('Random Forest Estimators'),
    dcc.Slider(id='hours', value=5, min=0, max=24, step=1),

html.Label('K neighbors'),
    dcc.Slider(id='rate', value=5, min=0, max=24, step=1),

html.Label('RandomForestClassifier Accuracy'),
    html.Div(id='amount'),

html.Label('KNN Accuracy'),
    html.Div(id='amount-per-week')
])

@app.callback(Output('amount', 'children'),
              [Input('hours', 'value')])
def compute_amount(hours):
    return steel_data_analysis.RandomForest(hours)

@app.callback(Output('amount-per-week', 'children'),
              [Input('rate', 'value')])
def compute_amount(rate):
    return steel_data_analysis.Kneighbors(rate)


if __name__ == '__main__':
    app.run_server(debug=True)
