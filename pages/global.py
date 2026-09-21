import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from predict_service import YosokuZonePredictor
from pages._shared_ui import create_heatmap

dash.register_page(__name__, path="/global", name="Global-Model")

try:
    predictor = YosokuZonePredictor()
    MODEL_OK = True
    ERR = ""
except Exception as e:
    predictor = None
    MODEL_OK = False
    ERR = str(e)

layout = dbc.Container([
    dbc.Row(dbc.Col([
        html.H3("Global-Model Prediction"),
        html.P("Predicting the penalty zone using a global model (without player data).", className="text-muted")
    ])),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Inputs (Global)"),
            dbc.CardBody([
                dbc.Alert(
                    f"Model not loaded: {ERR}" if not MODEL_OK else "Model loaded successfully.",
                    color="warning" if not MODEL_OK else "success"
                ),

                html.Label("Foot:"),
                dcc.RadioItems(
                    id="g-foot",
                    options=[{"label":"Right Foot", "value":"Right"}, {"label":"Left Foot", "value":"Left"}],
                    value="Right",
                    inline=True
                ),

                html.Label("Match time (min):", className="mt-3"),
                dcc.Slider(
                    id="g-time",
                    min=1, max=120, value=75, step=1,
                    marks={1:"1",45:"45",90:"90",120:"120"},
                    tooltip={"placement":"bottom", "always_visible": True}
                ),

                html.Label("Score difference:", className="mt-3"),
                dcc.Slider(
                    id="g-score",
                    min=-3, max=3, value=0, step=1,
                    marks={i:str(i) for i in range(-3,4)},
                    tooltip={"placement":"bottom", "always_visible": True}
                ),

                html.Label("Home/Away:", className="mt-3"),
                dcc.RadioItems(
                    id="g-homeaway",
                    options=[{"label":"Home", "value":1}, {"label":"Away", "value":0}],
                    value=1,
                    inline=True
                ),

                html.Label("Shootout penalty:", className="mt-3"),
                dcc.RadioItems(
                    id="g-shootout",
                    options=[{"label":"No (in-game)", "value":0}, {"label":"Yes (shootout)", "value":1}],
                    value=0,
                    inline=True
                ),

                dbc.Button("Predict", id="g-predict", className="mt-4 w-100", color="primary", size="lg"),
            ])
        ]), md=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Results"),
            dbc.CardBody([
                html.Div(id="g-meta"),
                dcc.Graph(id="g-heatmap", figure=create_heatmap(None)),
                html.Div(id="g-top")
            ])
        ]), md=8),
    ])
], fluid=True)


@dash.callback(
    [Output("g-heatmap", "figure"),
     Output("g-top", "children"),
     Output("g-meta", "children")],
    Input("g-predict", "n_clicks"),
    State("g-foot", "value"),
    State("g-time", "value"),
    State("g-score", "value"),
    State("g-homeaway", "value"),
    State("g-shootout", "value"),
)
def predict_global(n, foot, t, sd, ha, so):
    if n is None:
        return create_heatmap(None), "", ""

    if not MODEL_OK or predictor is None:
        return create_heatmap(None), "", dbc.Alert("Model not loaded. Train first.", color="warning")

    probs, pressure, meta = predictor.predict(
        mode="Global",
        player_name="",
        foot=foot,
        match_time=int(t),
        score_diff=int(sd),
        home_away=int(ha),
        is_shootout=int(so),
    )

    fig = create_heatmap(probs)
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    top_ui = html.Div([
        html.H6("Top 3 zones:"),
        html.Ol([html.Li(f"{z}: {p*100:.1f}%") for z, p in top3])
    ])

    meta_ui = dbc.Alert(
        [html.Div(f"Pressure index: {pressure:.2f} / 10"),
         html.Div("Used: global-only")],
        color="info"
    )
    return fig, top_ui, meta_ui