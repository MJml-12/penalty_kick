import plotly.graph_objects as go

ZONES = ["TL","TC","TR","ML","MC","MR","BL","BC","BR"]

def create_heatmap(probs):
    if probs is None:
        z = [[0,0,0],[0,0,0],[0,0,0]]
        text = [["TL","TC","TR"],["ML","MC","MR"],["BL","BC","BR"]]
    else:
        pct = {k: probs[k] * 100 for k in ZONES}
        z = [
            [pct["TL"], pct["TC"], pct["TR"]],
            [pct["ML"], pct["MC"], pct["MR"]],
            [pct["BL"], pct["BC"], pct["BR"]],
        ]
        text = [
            [f"TL<br>{pct['TL']:.1f}%", f"TC<br>{pct['TC']:.1f}%", f"TR<br>{pct['TR']:.1f}%"],
            [f"ML<br>{pct['ML']:.1f}%", f"MC<br>{pct['MC']:.1f}%", f"MR<br>{pct['MR']:.1f}%"],
            [f"BL<br>{pct['BL']:.1f}%", f"BC<br>{pct['BC']:.1f}%", f"BR<br>{pct['BR']:.1f}%"],
        ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        colorscale="RdYlGn",
        showscale=True,
        colorbar=dict(title="Probability (%)")
    ))
    fig.update_layout(
        title="Predicted Target Zone (3x3)",
        xaxis=dict(tickmode="array", tickvals=[0,1,2], ticktext=["Left","Center","Right"], side="top"),
        yaxis=dict(tickmode="array", tickvals=[0,1,2], ticktext=["Top","Middle","Bottom"], autorange="reversed"),
        height=420,
        margin=dict(l=80, r=80, t=70, b=50),
    )
    return fig