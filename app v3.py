# Import necessary libraries
import json
from dash import Dash, dcc, html, callback, Input, Output, State, dash_table
import base64
import io
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.utils import resample
# Import necessary libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define external stylesheets for Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize Dash app
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define layout for the app
app.layout = html.Div([
    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Button('Select Files', style={'background-color': 'blue', 'color': 'white'}),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    # Buttons for various actions
    html.Button('Show DataFrame Info', id='show-info-btn', n_clicks=0),
    html.Div(id='dataframe-info'),
    html.Button('Show Missing Values Info', id='show-missing-values-btn', n_clicks=0),
    html.Div(id='missing-values-info'),
    html.Button('Convert to Categorical', id='convert-to-categorical-button', n_clicks=0),
    dcc.Dropdown(id='column-to-convert', multi=True),
    html.Div(id='conversion-result'),
    html.Button('Show Summary Statistics', id='show-summary-btn', n_clicks=0),
    html.Div(id='summary-statistics'),
    html.Button('Show Visualizations', id='show-visualizations-btn', n_clicks=0),
    html.Div(id='visualization'),
    html.Div(id='heatmap'), 
    html.Button('Resample', id='resample-button', n_clicks=0),
    html.Div(id='resampled-data'),
    # DataTable to display resampled data
    html.Div(id='resampled-data-table'),
    # Layout section
    html.Div([
        dcc.Dropdown(id='y-variable'),
        # Dropdown for model selection
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Random Forest', 'value': 'RandomForestClassifier'},
                {'label': 'Support Vector Machine', 'value': 'SVC'},
                {'label': 'Logistic Regression', 'value': 'LogisticRegression'}
                # Add more models as needed
            ],
            value=[],  # Default selected model
            multi=True  # Allow multiple selections
        ),
        # Button to trigger model evaluation
        html.Button('Evaluate', id='evaluate-button', n_clicks=0),
        # Model performance metrics table
        html.Div(id='model-performance-metrics')
    ]),
    
    ])

# Function to parse uploaded contents
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

# Callback to update output data upload
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

# Callback to show DataFrame info
@app.callback(
    [Output('dataframe-info', 'children'),
     Output('missing-values-info', 'children')],
    Input('show-info-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_info(n_clicks, contents, filename):
    if n_clicks and contents is not None:
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

# Callback to update dropdown options for column conversion
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

# Callback to convert selected columns to categorical
@app.callback(
    Output('conversion-result', 'children'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def convert_to_categorical(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        return html.Div([
            html.H5('Conversion Result:'),
            html.P(f'Successfully converted {", ".join(selected_columns)} to categorical.')
        ])

# Callback to show summary statistics
@app.callback(
    Output('summary-statistics', 'children'),
    Input('show-summary-btn', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_summary_statistics(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        
        summary = df.describe(include='all').transpose()
        summary.insert(0, 'Column Name', summary.index)
        summary = summary.reset_index(drop=True)
        
        return html.Div([
            html.H5('Summary Statistics:'),
            dash_table.DataTable(data=summary.to_dict('records'), columns=[{"name": i, "id": i} for i in summary.columns])
        ])
        
# Callback to show visualizations
@app.callback(
    Output('visualization', 'children'),
    Input('show-visualizations-btn', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_visualizations(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
        
        figs = []
        if numerical_cols and categorical_cols:
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

# Callback to show heatmap
@app.callback(
    Output('heatmap', 'children'),
    Input('show-visualizations-btn', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_heatmap(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')

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
    
    
# Callback to update y variable dropdown for resampling
@app.callback(
    Output('y-variable', 'options'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_y_variable_dropdown(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if selected_columns:
            options = [{'label': col, 'value': col} for col in df.columns]
            return options
    return []

# Callback to resample data
@app.callback(
    [Output('resampled-data', 'children'),
     Output('resampled-data-table', 'children')],
    [Input('resample-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def resample_data_and_display_table(n_clicks, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        
        # Assuming 'Outcome' is the y variable column for resampling
        y_variable = 'Outcome'  # Change this according to your actual column name
        
        if y_variable in df.columns:
            df_majority = df[df[y_variable] == df[y_variable].value_counts().idxmax()]
            df_minority = df[df[y_variable] == df[y_variable].value_counts().idxmin()]

            df_minority_upsampled = resample(df_minority,
                                             replace=True,
                                             n_samples=df_majority.shape[0],
                                             random_state=123)

            df_resampled = pd.concat([df_majority, df_minority_upsampled])

            value_counts = df_resampled[y_variable].value_counts().reset_index()
            value_counts.columns = [y_variable, 'count']

            # Convert resampled data to DataTable
            resampled_table = dash_table.DataTable(
                id='resampled-data-table',
                columns=[{"name": i, "id": i} for i in value_counts.columns],
                data=value_counts.to_dict('records')
            )

            return value_counts.to_json(date_format='iso', orient='split'), resampled_table
    return None, None

@app.callback(
    Output('model-performance-metrics', 'children'),
    Input('evaluate-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('resampled-data-table', 'children')
)
def evaluate_models_and_display_metrics(n_clicks, selected_models, resampled_data_table):
    if n_clicks and selected_models and resampled_data_table:
        try:
            # Parse resampled data from DataTable
            resampled_df = pd.DataFrame(resampled_data_table[0]['props']['data'])

            # Assuming 'Outcome' is your target variable
            X = resampled_df.drop(columns=['Outcome'])
            y = resampled_df['Outcome']

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Preprocessing for numerical features
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Preprocessing for categorical features
            categorical_features = X.select_dtypes(include=['object']).columns
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Define models
            models = {
                'RandomForestClassifier': RandomForestClassifier(),
                'SVM': SVC(),
                'LogisticRegression': LogisticRegression()
                # Add more models as needed
            }

            # Create a pipeline for each selected model
            pipelines = {}
            for name, model in models.items():
                if name in selected_models:
                    pipelines[name] = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])

            # Define an empty dictionary to store evaluation metrics
            results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

            # Evaluate each model and store the results
            for name, pipeline in pipelines.items():
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                results['Model'].append(name)
                results['Accuracy'].append(accuracy)
                results['Precision'].append(precision)
                results['Recall'].append(recall)
                results['F1-score'].append(f1)

            # Create a DataFrame from the results dictionary
            results_df = pd.DataFrame(results)

            # Convert DataFrame to DataTable
            model_metrics_table = dash_table.DataTable(
                id='model-performance-metrics-table',
                columns=[{"name": i, "id": i} for i in results_df.columns],
                data=results_df.to_dict('records')
            )

            return model_metrics_table
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return html.Div("An error occurred while evaluating models.")
    return html.Div("Please select models and resampled data, then click Evaluate.")


if __name__ == '__main__':
    app.run_server(debug=True)