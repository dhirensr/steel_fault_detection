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
    html.Div(id='amount-per-week'),
html.H3('Heatmap-RandomForestClassifier'),
    html.Div(id='confusion_matrix'),
html.H3('Heatmap- KNN Confusion'),
    html.Div(id='confusion_matrix_1'),

html.H3('Prediction for P1'),
    dcc.Graph(
        id='graph-2-tabs',
        figure={
            'data': [{
                'x': [1,2,3],
                'y': [4,5,7],
                'type': 'scatter',
                "name" : "temperature"
            }]
        }
    ),

html.H3('Prediction for P1'),
    dcc.Graph(
                id='graph-3-tabs',
                figure={
                    'data': [{
                        'x': [1,2,3],
                        'y': [4,5,7],
                        'type': 'scatter',
                        "name" : "temperature"
                    }]
                }
            )])

@app.callback(Output('amount', 'children'),
              [Input('hours', 'value')])
def compute_amount(hours):
    return steel_data_analysis.RandomForest(hours)[1]

@app.callback(Output('confusion_matrix_1', 'children'),
              [Input('rate', 'value')])
def heatmap_randomforest(rate):
    return html.Div([dcc.Graph(
        id='graph-1-tabs',
        figure={
            'data': [{
                'z':  steel_data_analysis.Kneighbors(rate)[0],
                'text': [
                    ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                    ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults']
                ],
                'type': 'heatmap',
                "name" : "temperature"
            }]
        }
    )])

@app.callback(Output('confusion_matrix', 'children'),
              [Input('rate', 'value')])
def heatmap_knn(rate):
    return html.Div([dcc.Graph(
        id='graph-2-tabs',
        figure={
            'data': [{
                'z':  steel_data_analysis.Kneighbors(rate)[0],
                'text': [
                    ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                    ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults']
                ],
                'type': 'heatmap',
                "name" : "temperature"
            }]
        }
    )])

@app.callback(Output('amount-per-week', 'children'),
              [Input('rate', 'value')])
def compute_amount(rate):
    return steel_data_analysis.Kneighbors(rate)[1]


if __name__ == '__main__':
    app.run_server(debug=True)
