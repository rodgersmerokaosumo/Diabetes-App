import json
import base64
import io
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        print(e)
        return None

def create_data_upload_output(df, filename):
    if df is not None:
        sliced_df = pd.concat([df.head(4), df.tail(4)])
        return html.Div([
            html.H5(f'File uploaded: {filename}'),
            dash_table.DataTable(data=sliced_df.to_dict('records')),
        ])

def create_tabs():
    return html.Div([
        dcc.Tabs(id="tabs", value='tab1', children=[
            dcc.Tab(label='Data Exploration', value='tab1'),
            dcc.Tab(label='Model Evaluation', value='tab2'),
            dcc.Tab(label='Model Tuning', value='tab3'),
            dcc.Tab(label='Prediction', value='tab4'),
        ]),
        html.Div(id='tabs-content')
    ])

def create_tab_content(tab):
    if tab == 'tab1':
        return create_data_exploration_tab()
    elif tab == 'tab2':
        return html.Div(id='model-evaluation')
    elif tab == 'tab3':
        return html.Div([
            html.H3('Model Tuning Tab'),
            # Add model tuning components here
        ])
    elif tab == 'tab4':
        return html.Div([
            html.H3('Prediction Tab'),
            # Add prediction components here
        ])

def create_data_exploration_tab():
    return html.Div([
        html.Button('Show DataFrame Info', id='show-info-btn', n_clicks=0),
        html.Div(id='dataframe-info'),
        html.Button('Show Missing Values Info', id='show-missing-values-btn', n_clicks=0),
        html.Div(id='missing-values-info'),
        html.Hr(),
        dcc.Dropdown(id='column-to-convert', multi=True),
        html.Button('Convert to Categorical', id='convert-to-categorical-button', n_clicks=0),
        html.Div(id='conversion-result'),
        html.Button('Show Summary Statistics', id='show-summary-btn', n_clicks=0),
        html.Div(id='summary-statistics'),
        html.Button('Show Visualizations', id='show-visualizations-btn', n_clicks=0),
        html.Div(id='visualization'),
        html.Div(id='heatmap'), 
        html.Button('Resample', id='resample-button', n_clicks=0),
        dcc.Dropdown(id='y-variable'),
        html.Div(id='resampled-data'),
        html.Div(id='train-test-split'),
    ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    df = parse_contents(contents, filename)
    return create_data_upload_output(df, filename)

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
)
def render_content(tab):
    return create_tab_content(tab)

@app.callback(
    Output('dataframe-info', 'children'),
    Output('missing-values-info', 'children'),
    Input('show-info-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_info(n_clicks, contents, filename):
    if n_clicks is not None and n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_string = buffer.getvalue()
            missing_values = df.isnull().sum()
            missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values'])
            return (html.Div([
                        html.H5('DataFrame Info:'),
                        html.Pre(info_string)
                    ]),
                    html.Div([
                        html.H5('Missing Values Info:'),
                        dash_table.DataTable(data=missing_values_df.to_dict('records'), columns=[{"name": i, "id": i} for i in missing_values_df.columns])
                    ]))
    return html.Div(), html.Div()

@app.callback(
    Output('column-to-convert', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        return [{'label': i, 'value': i} for i in df.columns]
    return []

@app.callback(
    Output('conversion-result', 'children'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def convert_to_categorical(n_clicks, selected_columns, contents, filename):
    if n_clicks is not None and n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        return html.Div([
            html.H5('Conversion Result:'),
            html.P(f'Successfully converted {", ".join(selected_columns)} to categorical.')
        ])

@app.callback(
    Output('summary-statistics', 'children'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_summary_statistics(n_clicks, selected_columns, contents, filename):
    if n_clicks is not None and n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        
        # Compute summary statistics
        summary = df.describe(include='all').transpose()
        summary.insert(0, 'Column Name', summary.index)
        summary = summary.reset_index(drop=True)
        
        return html.Div([
            html.H5('Summary Statistics:'),
            dash_table.DataTable(data=summary.to_dict('records'), columns=[{"name": i, "id": i} for i in summary.columns])
        ])
        
@app.callback(
    Output('visualization', 'children'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_visualizations(n_clicks, selected_columns, contents, filename):
    if n_clicks is not None and n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        
        # Create visualizations
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Convert to list
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()  # Convert to list
        
        figs = []
        if numerical_cols and categorical_cols:  # Check both lists are not empty
            for col in numerical_cols:
                fig = px.histogram(df, x=col)
                figs.append(html.Div(dcc.Graph(figure=fig)))
            
            for col in categorical_cols:
                value_counts_df = df[col].value_counts().reset_index()
                value_counts_df.columns = [col, 'count']
                fig = px.bar(value_counts_df, x=col, y='count')
                figs.append(html.Div(dcc.Graph(figure=fig)))
                
            for num_col in numerical_cols:
                for cat_col in categorical_cols:
                    fig = px.histogram(df, x=num_col, color=cat_col, marginal="violin", hover_data=df.columns)
                    figs.append(html.Div(dcc.Graph(figure=fig)))
        else:
            figs.append(html.Div([
                html.H5('Visualizations:'),
                html.P('No numerical and categorical column pairs found for visualization.')
            ]))
        
        return html.Div([
            html.H5('Visualizations:'),
            *figs
        ])

        
        
@app.callback(
    Output('heatmap', 'children'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_heatmap(n_clicks, selected_columns, contents, filename):
    if n_clicks is not None and n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')

        # Create heatmap
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numerical_cols:
            corr = df[numerical_cols].corr().round(2)
            heatmap = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Viridis')
            return html.Div([
                html.H5('Heatmap:'),
                dcc.Graph(figure=heatmap)
            ])
        else:
            return html.Div([
                html.H5('Heatmap:'),
                html.P('No numerical columns found for heatmap visualization.')
            ])

    return html.Div([
        html.H5('Heatmap:'),
        html.P('Upload a file and select columns to generate heatmap.')
    ])
    
    
@app.callback(
    Output('y-variable', 'options'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_y_variable_dropdown(n_clicks, selected_columns, contents, filename):
    if n_clicks is not None and n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if selected_columns:
            options = [{'label': col, 'value': col} for col in df.columns]
            return options
    return []


    
@app.callback(
    Output('resampled-data', 'children'),
    Input('resample-button', 'n_clicks'),
    State('y-variable', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def resample_data(n_clicks, y_variable, contents, filename):
    if n_clicks is not None and n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)

        # Resampling
        if y_variable is not None:
            # Separate majority and minority classes
            df_majority = df[df[y_variable] == df[y_variable].value_counts().idxmax()]
            df_minority = df[df[y_variable] == df[y_variable].value_counts().idxmin()]

            # Upsample minority class
            df_minority_upsampled = resample(df_minority,
                                             replace=True,  # sample with replacement
                                             n_samples=df_majority.shape[0],  # to match majority class
                                             random_state=123)  # reproducible results

            # Combine majority class with upsampled minority class
            df_resampled = pd.concat([df_majority, df_minority_upsampled])

            # Compute value counts for Y-variable after resampling
            value_counts = df_resampled[y_variable].value_counts().reset_index()
            value_counts.columns = [y_variable, 'count']

            return value_counts.to_json(date_format='iso', orient='split')

    return None


if __name__ == '__main__':
    app.run_server(debug=True)
