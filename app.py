# Source: Board of Governors of the Federal Reserve System (US), via FRED
import os

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

# 🔑 FRED API 키 입력
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# ✅ 지표 목록 및 이름 매핑
indicators = {
    "RRPONTSYD": "Reverse Repo (RRP)",
    "WDTGAL": "TGA (Treasury General Account)",
    "M2SL": "M2 Money Supply",
    "FEDFUNDS": "Federal Funds Rate",
    "BUSLOANS": "C&I Loans",
    "BAMLH0A0HYM2": "High-Yield Spread",
}


# 📊 데이터 불러오기
def fetch_data(series_id):
    data = fred.get_series(series_id)
    df = data.reset_index()
    df.columns = ["Date", "Value"]
    df["Series"] = indicators[series_id]
    return df


# 🔁 모든 지표 데이터 통합
all_data = pd.concat([fetch_data(sid) for sid in indicators.keys()])
all_data.dropna(inplace=True)

# 🎨 Dash 앱 구성
app = Dash(__name__)
app.title = "Liquidity Dashboard"

app.layout = html.Div(
    [
        html.H1("📊 US Liquidity Composite Dashboard", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="indicator-dropdown",
            options=[
                {"label": name, "value": name} for name in all_data["Series"].unique()
            ],
            value="Reverse Repo (RRP)",
        ),
        dcc.Graph(id="indicator-graph"),
    ]
)


@app.callback(
    dash.dependencies.Output("indicator-graph", "figure"),
    [dash.dependencies.Input("indicator-dropdown", "value")],
)
def update_graph(selected_indicator):
    df = all_data[all_data["Series"] == selected_indicator]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Value"], mode="lines", name=selected_indicator)
    )
    fig.update_layout(
        title=selected_indicator, xaxis_title="Date", yaxis_title="Value", height=500
    )
    return fig


# ▶ 실행
if __name__ == "__main__":
    app.run(debug=True)
