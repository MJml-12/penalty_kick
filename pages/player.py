import json
import os

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from predict_service import YosokuZonePredictor
from pages._shared_ui import create_heatmap
from pages.data_utils import PLAYER_FOOT_MAP

dash.register_page(__name__, path="/player", name="Player-Based")

try:
    predictor = YosokuZonePredictor()
    MODEL_OK = True
    ERR = ""
except Exception as e:
    predictor = None
    MODEL_OK = False
    ERR = str(e)

PLAYER_OPTIONS = []
if os.path.exists("models/player_priors.json"):
    with open("models/player_priors.json", "r", encoding="utf-8") as f:
        priors = json.load(f)
    names = sorted(priors.keys())
    PLAYER_OPTIONS = [{"label": n, "value": n} for n in names]

layout = dbc.Container([
    dbc.Row(dbc.Col([
        html.H3("Player-Based Prediction"),
        html.P("Predicting the penalty zone with a player prior option (searchable dropdown).", className="text-muted")
    ])),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Inputs (Player)"),
            dbc.CardBody([
                dbc.Alert(
                    f"Model not loaded: {ERR}" if not MODEL_OK else "Model loaded successfully.",
                    color="warning" if not MODEL_OK else "success"
                ),

                html.Label("Mode:"),
                dcc.RadioItems(
                    id="p-mode",
                    options=[
                        {"label":"Auto (player prior + global)", "value":"Auto"},
                        {"label":"Player-only (prior only)", "value":"Player"},
                        {"label":"Global-only (ignore player)", "value":"Global"},
                    ],
                    value="Auto"
                ),

                html.Hr(),

                html.Label("Player name:"),
                dcc.Dropdown(
                    id="p-player",
                    options=PLAYER_OPTIONS,
                    value=None,
                    searchable=True,
                    placeholder="Type to search player name...",
                    clearable=True,
                ),

                html.Label("Foot:", className="mt-3"),
                html.Small("Automatically detected from historical data (editable).",className="text-muted"),
                dcc.RadioItems(
                    id="p-foot",
                    options=[{"label":"Right Foot", "value":"Right"}, {"label":"Left Foot", "value":"Left"}],
                    value="Right",
                    inline=True
                ),

                html.Label("Match time (min):", className="mt-3"),
                dcc.Slider(
                    id="p-time",
                    min=1, max=120, value=75, step=1,
                    marks={1:"1",45:"45",90:"90",120:"120"},
                    tooltip={"placement":"bottom", "always_visible": True}
                ),

                html.Label("Score difference:", className="mt-3"),
                dcc.Slider(
                    id="p-score",
                    min=-3, max=3, value=0, step=1,
                    marks={i:str(i) for i in range(-3,4)},
                    tooltip={"placement":"bottom", "always_visible": True}
                ),

                html.Label("Home/Away:", className="mt-3"),
                dcc.RadioItems(
                    id="p-homeaway",
                    options=[{"label":"Home", "value":1}, {"label":"Away", "value":0}],
                    value=1,
                    inline=True
                ),

                html.Label("Shootout penalty:", className="mt-3"),
                dcc.RadioItems(
                    id="p-shootout",
                    options=[{"label":"No (in-game)", "value":0}, {"label":"Yes (shootout)", "value":1}],
                    value=0,
                    inline=True
                ),

                dbc.Button("Predict", id="p-predict", className="mt-4 w-100", color="primary", size="lg"),
            ])
        ]), md=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Results"),
            dbc.CardBody([
                html.Div(id="p-meta"),
                dcc.Graph(id="p-heatmap", figure=create_heatmap(None)),
                html.Div(id="p-top")
            ])
        ]), md=8),
    ])
], fluid=True)


@dash.callback(
    [Output("p-heatmap", "figure"),
     Output("p-top", "children"),
     Output("p-meta", "children")],
    Input("p-predict", "n_clicks"),
    State("p-mode", "value"),
    State("p-player", "value"),
    State("p-foot", "value"),
    State("p-time", "value"),
    State("p-score", "value"),
    State("p-homeaway", "value"),
    State("p-shootout", "value"),
)
def predict_player(n, mode, player, foot, t, sd, ha, so):
    if n is None:
        return create_heatmap(None), "", ""

    if not MODEL_OK or predictor is None:
        return create_heatmap(None), "", dbc.Alert("Model not loaded. Train first.", color="warning")

    probs, pressure, meta = predictor.predict(
        mode=mode,
        player_name=(player or "").strip(),
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
        [
            html.Div(f"Pressure index: {pressure:.2f} / 10"),
            html.Div(f"Used: {meta['used']} | alpha: {meta['alpha']:.2f} | player_k: {meta['player_k']}"),
        ],
        color="info"
    )
    return fig, top_ui, meta_ui

@dash.callback(
    Output("p-foot", "value"),
    Input("p-player", "value"),
    State("p-foot", "value"),
    prevent_initial_call=True
)
def auto_detect_foot(player_name, current_foot):
    if not player_name:
        return dash.no_update

    foot = PLAYER_FOOT_MAP.get(player_name)

    if foot is None:
        return dash.no_update

    if foot.lower().startswith("r"):
        return "Right"
    elif foot.lower().startswith("l"):
        return "Left"

    return dash.no_update