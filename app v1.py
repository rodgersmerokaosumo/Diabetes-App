import json
from dash import Dash, dcc, html, callback, Input, Output, State, dash_table
import base64
import io
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.utils import resample  # Importing resample function

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Button('Select Files', style={'background-color': 'blue', 'color': 'white'}),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Button('Show DataFrame Info', id='show-info-btn', n_clicks=0),
    html.Div(id='dataframe-info'),
    html.Button('Show Missing Values Info', id='show-missing-values-btn', n_clicks=0),
    html.Div(id='missing-values-info'),
    html.Hr(),
    # Add dropdown and button for converting columns to categorical
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


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df


@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        sliced_df = pd.concat([df.head(4), df.tail(4)])
        return html.Div([
            html.H5(f'File uploaded: {filename}'),
            dash_table.DataTable(data=sliced_df.to_dict('records')),
        ])


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
    State('upload-data', 'data')
)
def show_visualizations(n_clicks, selected_columns, data):
    if not n_clicks or not data:
        return []

    df = pd.read_json(data, orient='split')
    if not selected_columns:
        return []

    figs = []
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()

    if numerical_cols and categorical_cols:
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
