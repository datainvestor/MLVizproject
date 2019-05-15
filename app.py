import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


df=pd.read_csv("Iris.csv")
df=df.drop(['Id'], axis=1)
X = np.array(df.ix[:, 0:4])
y = np.array(df['Species']) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(children='ML Visualization App'),

    html.Div(children='''
        A simple web application made for Data Analytics class by Aleksander Trinh.
    ''',style={'fontSize': 19}),

    html.Div(children='''
        Using Iris dataset and K nearest neighbours algorithm, visualize
        how different number of neighbours affects clusters.
        Choose different number of Neighbours on the slider at the bottom.
    ''',style={'fontSize': 16}),

    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='slider',
        min=1,
        max=80,
        value=5,
        marks={i: '{} Neighbours'.format(i) for i in range(1,80,10)},
        step=10
    )
])


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('slider', 'value')])
def update_figure(input_value):
   
    knn = KNeighborsClassifier(n_neighbors=int(input_value))
  
    # fitting the model
    knn.fit(X_train, y_train)
    #  predict the response
    pred = knn.predict(X_test)
    dfp=pd.DataFrame(X_test)
    dfp.columns = ['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    dfp["PClass"]=pred

    return {
        'data': [
        {
            'x': dfp[dfp['PClass']==sp]['SepalLengthCm'],
            'y': dfp[dfp['PClass']==sp]['SepalWidthCm'],
            'name': sp, 'mode': 'markers',
        } for sp in ["Iris-setosa", 'Iris-versicolor', "Iris-virginica"]
        ],
        'layout': dict(
            xaxis={
                'title': 'Sepal Length in Cm',
                'range': [4, 8]
            },
            yaxis={
                'title': 'Sepal Width in Cm',
                'range': [2, 5]
            },
            legend={'x': 0, 'y': 1},
            hovermode='closest',
            transition={
                'duration': 500,
                'easing': 'cubic-in-out'
            }
        )
    }
    

if __name__ == '__main__':
    app.run_server(debug=True)