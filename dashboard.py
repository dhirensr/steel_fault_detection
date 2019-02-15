import dash
from dash.dependencies import Input, Output
import dash_html_components as html


import dash_core_components as dcc
import steel_data_analysis
import datetime

cnf,accuracy,test_count,pred_count=[[0,0]],0,[0,0,0],[0,0,0]
cnf_knn,accuracy_knn,test_count_knn,pred_count_knn=[[0,0]],0,[0,0,0],[0,0,0]


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Label('Random Forest Estimators'),
    dcc.Slider(id='hours', value=5, min=0, max=24, step=1),

html.Label('K neighbors'),
    dcc.Slider(id='rate', value=5, min=0, max=24, step=1),

html.Div(id='amount-dummy'),
html.Div(id='amount_dummy'),

html.Label('Select Attributes'),
    dcc.Checklist(id='attributes',
    options=[
        {'label': 'X_Minimum', 'value': 'V1'},
        {'label': 'X_Maximum', 'value': 'V2'},
        {'label': 'Y_Minimum ', 'value': 'V3'},
        {'label': 'Y_Maximum ', 'value': 'V4'},
        {'label': "Pixels_Areas", 'value': 'V5'},
        {'label': 'X_Perimeter', 'value': 'V6'},
        {'label': 'Y_Perimeter', 'value': 'V7'},
        {'label': 'Sum_of_Luminosity ', 'value': 'V8'},
        {'label': 'Minimum_of_Luminosity ', 'value': 'V9'},
        {'label': 'Maximum_of_Luminosity', 'value': 'V10'},
        {'label': 'Length_of_Conveyer', 'value': 'V11'},
        {'label': 'TypeOfSteel_A300 ', 'value': 'V12'},
        {'label': 'TypeOfSteel_A400', 'value': 'V13'},
        {'label': 'Steel_Plate_Thickness', 'value': 'V14'},
        {'label': 'Edges_Index', 'value': 'V15'},
        {'label': 'Empty_Index', 'value': 'V16'},
        {'label': 'Square_Index', 'value': 'V17'},
        {'label': 'Outside_X_Index', 'value': 'V18'},
        {'label': 'Edges_X_Index', 'value': 'V19'},
        {'label': 'Edges_Y_Index', 'value': 'V20'},
        {'label': 'Outside_Global_Index', 'value': 'V21'},
        {'label': 'LogOfAreas', 'value': 'V22'},
        {'label': 'Log_X_Index', 'value': 'V23'},
        {'label': "Log_Y_Index", 'value': 'V24'},
        {'label': 'Orientation_Index ', 'value': 'V25'},
        {'label': 'Luminosity_Index ', 'value': 'V26'},
        {'label': 'SigmoidOfAreas', 'value': 'V27'}

    ],
    values=['V1', 'V2']
    ),
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

@app.callback(Output('amount-dummy', 'children'),
              [Input('attributes', 'values'),Input('hours', 'value')])
def compute_amount(attributes,hours):
    global cnf,accuracy,test_count,pred_count
    cnf,accuracy,test_count,pred_count=steel_data_analysis.RandomForest(attributes,hours)
    return ""

@app.callback(Output('amount_dummy', 'children'),
              [Input('attributes', 'values'),Input('rate', 'value')])
def compute_amount(attributes,rate):
    global cnf_knn,accuracy_knn,test_count_knn,pred_count_knn
    cnf_knn,accuracy_knn,test_count_knn,pred_count_knn=steel_data_analysis.Kneighbors(attributes,rate)
    return ""


@app.callback(Output('amount-per-week', 'children'),
              [Input('attributes', 'values'),Input('rate', 'value')])
def compute_amount(attributes,rate):
    global cnf_knn,accuracy_knn,test_count_knn,pred_count_knn
    return accuracy_knn

@app.callback(Output('amount', 'children'),
              [Input('attributes', 'values'),Input('hours', 'value')])
def compute_amount(attributes,hours):
    global cnf,accuracy,test_count,pred_count
    return accuracy



@app.callback(Output('confusion_matrix', 'children'),
              [Input('attributes', 'values'),Input('hours', 'value')])
def heatmap_randomforest(attributes,hours):
    global cnf,accuracy,test_count,pred_count
    return html.Div([dcc.Graph(
        id='graph-1-tabs',
        figure={
            'data': [{
                'z':  cnf,
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'y' :['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'type': 'heatmap',
                "name" : "temperature"
            }]
        }
    )])

@app.callback(Output('confusion_matrix_1', 'children'),
              [Input('attributes', 'values'),Input('rate', 'value')])
def heatmap_knn(attributes,rate):
    global cnf_knn,accuracy_knn,test_count_knn,pred_count_knn
    return html.Div([dcc.Graph(
        id='graph-2-tabs',
        figure={
            'data': [{
                'z':  cnf_knn,
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'y' :['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'type': 'heatmap',
                "name" : "temperature-1"
            }]
        }
    )])


@app.callback(Output('TR_RF', 'children'),
              [Input('attributes', 'values'),Input('hours', 'value')])
def bar_randomforest_test(attributes,rate):
    global cnf,accuracy,test_count,pred_count
    return html.Div([dcc.Graph(
        id='graph-3-tabs',
        figure={
            'data': [{
                'y':  test_count,
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],

                'type': 'bar',
                "name" : "Test Results"
            }]
        }
    )])


@app.callback(Output('PR_RF', 'children'),
              [Input('attributes', 'values'),Input('hours', 'value')])
def bar_randomforest_predicted(attributes,rate):
    global cnf,accuracy,test_count,pred_count
    return html.Div([dcc.Graph(
        id='graph-4-tabs',
        figure={
            'data': [{
                'y':  pred_count,
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],

                'type': 'bar',
                "name" : "Test Results"
            }]
        }
    )])

@app.callback(Output('TR_KNN', 'children'),
              [Input('attributes', 'values'),Input('rate', 'value')])
def bar_knn_test(attributes,rate):
    global cnf_knn,accuracy_knn,test_count_knn,pred_count_knn
    return html.Div([dcc.Graph(
        id='graph-5-tabs',
        figure={
            'data': [{
                'y':  test_count_knn,
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'type': 'bar',
                "name" : "Test Results"
            }]
        }
    )])


@app.callback(Output('PR_KNN', 'children'),
              [Input('attributes', 'values'),Input('rate', 'value')])
def bar_knn_predicted(attributes,rate):
    global cnf_knn,accuracy_knn,test_count_knn,pred_count_knn
    return html.Div([dcc.Graph(
        id='graph-6-tabs',
        figure={
            'data': [{
                'y':  pred_count_knn,
                'x': ['Pastry', 'Z-scratch', 'K_Scratch',"Stains",'Dirtiness','Bumps','Other_faults'],
                'type': 'bar',
                "name" : "Predicted Results"
            }]
        }
    )])


if __name__ == '__main__':
    app.run_server(debug=True)
