"""
Yosoku System v3 - Multi-page Dashboard entrypoint
Run: python app.py
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Penalty Zone Predictor"

navbar = dbc.NavbarSimple(
    brand="⚽ Penalty Predictor",
    brand_href="/",
    color="dark",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Global-Model", href="/global")),
        dbc.NavItem(dbc.NavLink("Player-Based", href="/player")),
    ],
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        navbar,
        html.Div(dash.page_container, className="mt-4"),
    ],
    fluid=True,
)

if __name__ == "__main__":
    print("📍 http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)