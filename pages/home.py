import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/", name="Home")

layout = dbc.Container(
    [
        html.H3("Welcome to Penalty Predictor"),
        html.P(
            "Use the navigation bar to open the Global-Model or Player-Based prediction pages.",
            className="text-muted",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Go to Global-Model Prediction", href="/global", color="primary", className="w-100",style={'backgroundcolor':'#6C8EAD'}),
                    md=6,
                ),
                dbc.Col(
                    dbc.Button("Go to Player-Based Prediction", href="/player", color="secondary", className="w-100"),
                    md=6,
                ),
            ],
            className="mt-3",
        ),
        # dbc.Alert(
        #     "Note: If you previously saw a 'Not Found' page on first load, "
        #     "it was because the root route '/' had no registered page. "
        #     "This Home page fixes that.",
        #     color="info",
        #     className="mt-4",
        # ),
    ],
    fluid=True,
)