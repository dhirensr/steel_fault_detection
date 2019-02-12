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
html.Label('Heatmap-RandomForestClassifier'),
    html.Div(id='confusion_matrix'),
html.Label('Heatmap- KNN Confusion'),
    html.Div(id='confusion_matrix_1'),

###################
html.Label('Test_Results-RandomForestClassifier Bar'),
    html.Div(id='TR_RF'),
html.Label('Predicted_Results-RandomForestClassifier Bar'),
    html.Div(id='PR_RF'),



html.Label('Test_Results-KNN Bar'),
    html.Div(id='TR_KNN'),
html.Label('Predicted_Results-KNN Bar'),
    html.Div(id='PR_KNN')

######################



])

@app.callback(Output('amount', 'children'),
              [Input('hours', 'value')])
def compute_amount(hours):
    return steel_data_analysis.RandomForest(hours)[1]

@app.callback(Output('confusion_matrix', 'children'),
              [Input('rate', 'value')])
def heatmap_randomforest(rate):
    return html.Div([dcc.Graph(
        id='graph-1-tabs',
        figure={
            'data': [{
                'z':  steel_data_analysis.RandomForest(rate)[0],
                'text': [
                    ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                    ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults']
                ],
                'type': 'heatmap',
                "name" : "temperature"
            }]
        }
    )])

@app.callback(Output('confusion_matrix_1', 'children'),
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


@app.callback(Output('TR_RF', 'children'),
              [Input('rate', 'value')])
def bar_randomforest_test(rate):
    return html.Div([dcc.Graph(
        id='graph-3-tabs',
        figure={
            'data': [{
                'y':  steel_data_analysis.RandomForest(rate)[2],
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],

                'type': 'bar',
                "name" : "Test Results"
            }]
        }
    )])


@app.callback(Output('PR_RF', 'children'),
              [Input('rate', 'value')])
def bar_randomforest_predicted(rate):
    return html.Div([dcc.Graph(
        id='graph-4-tabs',
        figure={
            'data': [{
                'y':  steel_data_analysis.RandomForest(rate)[3],
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],

                'type': 'bar',
                "name" : "Predicted Results"
            }]
        }
    )])

@app.callback(Output('TR_KNN', 'children'),
              [Input('rate', 'value')])
def bar_knn_test(rate):
    return html.Div([dcc.Graph(
        id='graph-5-tabs',
        figure={
            'data': [{
                'y':  steel_data_analysis.Kneighbors(rate)[2],
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'type': 'bar',
                "name" : "Test Results"
            }]
        }
    )])


@app.callback(Output('PR_KNN', 'children'),
              [Input('rate', 'value')])
def bar_knn_predicted(rate):
    return html.Div([dcc.Graph(
        id='graph-6-tabs',
        figure={
            'data': [{
                'y':  steel_data_analysis.Kneighbors(rate)[3],
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'type': 'bar',
                "name" : "Predicted Results"
            }]
        }
    )])

@app.callback(Output('amount-per-week', 'children'),
              [Input('rate', 'value')])
def compute_amount(rate):
    return steel_data_analysis.Kneighbors(rate)[1]


if __name__ == '__main__':
    app.run_server(debug=True)
