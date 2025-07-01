# Source: Board of Governors of the Federal Reserve System (US), via FRED

import os

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.preprocessing import MinMaxScaler

# 🔑 API 키 불러오기
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# ✅ 지표 목록 및 매핑
indicators = {
    "RRPONTSYD": "Reverse Repo (RRP)",
    "WDTGAL": "TGA (Treasury General Account)",
    "M2SL": "M2 Money Supply",
    "FEDFUNDS": "Federal Funds Rate",
    "BUSLOANS": "C&I Loans",
    "BAMLH0A0HYM2": "High-Yield Spread",
}

# 📝 한글 설명
indicator_descriptions = {
    "Reverse Repo (RRP)": "역레포는 연준이 초과 유동성을 흡수할 때 사용하는 단기 자금 운용 수단입니다. 수치가 높을수록 시장에 돈이 많이 남아 있다는 뜻입니다.",
    "TGA (Treasury General Account)": "재무부 일반 계정은 미 재무부가 연준에 보유한 예산 잔액을 나타냅니다. 유동성 흡수/공급에 영향을 줍니다.",
    "M2 Money Supply": "M2는 광의 통화량으로, 현금, 요구불 예금, 저축성 예금 등을 포함합니다. 전체 유동성 흐름을 보여줍니다.",
    "Federal Funds Rate": "연방기금금리는 미국 은행 간 초단기 금리로, 연준의 통화 정책 방향을 반영합니다.",
    "C&I Loans": "기업 대출(C&I Loans)은 은행이 기업에 빌려준 자금으로, 기업의 신용 수요와 경기 활력을 보여줍니다.",
    "High-Yield Spread": "하이일드 스프레드는 고위험 회사채와 국채의 금리 차이를 말하며, 금융시장의 스트레스 수준을 반영합니다.",
}


# 📊 FRED 데이터 불러오기
def fetch_data(series_id):
    data = fred.get_series(series_id)
    df = data.reset_index()
    df.columns = ["Date", "Value"]
    df["Series"] = indicators[series_id]
    return df


# 🔁 모든 지표 통합
all_data = pd.concat([fetch_data(sid) for sid in indicators.keys()])
all_data.dropna(inplace=True)


# ⚖️ 지표별 가중치 설정
indicator_weights = {
    "Reverse Repo (RRP)": 0.25,  # 높은 가중치 - 유동성 흡수 직접 지표
    "M2 Money Supply": 0.20,  # 높은 가중치 - 전체 유동성 규모
    "Federal Funds Rate": 0.20,  # 높은 가중치 - 연준 정책금리
    "TGA (Treasury General Account)": 0.15,  # 중간 가중치 - 유동성 변동 요인
    "C&I Loans": 0.10,  # 중간 가중치 - 신용 확장 지표
    "High-Yield Spread": 0.10,  # 중간 가중치 - 시장 스트레스 지표
}


# 📐 유동성 점수 계산 함수 (가중치 적용)
def calculate_liquidity_score(df):
    score_dict = {}
    scaler = MinMaxScaler()

    for series_name in df["Series"].unique():
        df_sub = df[df["Series"] == series_name].sort_values("Date")
        values = df_sub["Value"].values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        latest = scaled[-1][0]

        # 역방향 지표 (값이 낮을수록 유동성이 많은 것)
        if series_name in [
            "Reverse Repo (RRP)",
            "TGA (Treasury General Account)",
            "Federal Funds Rate",
            "High-Yield Spread",
        ]:
            latest = 1 - latest

        score_dict[series_name] = latest

    # 가중 평균 계산
    weighted_score = sum(
        score_dict[series] * indicator_weights.get(series, 0)
        for series in score_dict.keys()
    )

    return round(weighted_score * 100, 1)


# ✅ 유동성 점수 계산
liquidity_score = calculate_liquidity_score(all_data)


# 해석 메시지
def get_score_message(score):
    if score >= 80:
        return "🟢 유동성이 매우 풍부한 상태입니다."
    elif score >= 60:
        return "🟡 유동성이 비교적 충분한 편입니다."
    elif score >= 40:
        return "🟠 유동성이 중간 수준이며 주의가 필요합니다."
    else:
        return "🔴 유동성이 부족한 상태입니다. 긴축 국면일 수 있습니다."


# 🎨 Dash 앱 구성
app = Dash(__name__)
app.title = "Liquidity Dashboard"

app.layout = html.Div(
    [
        html.H1("📊 US Liquidity Composite Dashboard", style={"textAlign": "center"}),
        html.H2(
            f"💧 현재 유동성 점수: {liquidity_score} / 100",
            style={"textAlign": "center", "color": "#1a73e8"},
        ),
        html.P(
            get_score_message(liquidity_score),
            style={"textAlign": "center", "fontSize": "18px"},
        ),
        # 가중치 정보 섹션
        html.Div(
            [
                html.H3(
                    "⚖️ 지표별 가중치",
                    style={"textAlign": "center", "marginTop": "30px"},
                ),
                html.Div(
                    [
                        html.P(
                            f"{indicator}: {weight*100:.0f}%",
                            style={
                                "display": "inline-block",
                                "margin": "5px 15px",
                                "padding": "5px 10px",
                                "backgroundColor": "#f0f0f0",
                                "borderRadius": "5px",
                            },
                        )
                        for indicator, weight in indicator_weights.items()
                    ],
                    style={"textAlign": "center"},
                ),
            ]
        ),
        dcc.Dropdown(
            id="indicator-dropdown",
            options=[
                {"label": name, "value": name} for name in all_data["Series"].unique()
            ],
            value="Reverse Repo (RRP)",
            style={"width": "60%", "margin": "20px auto"},
        ),
        html.Div(
            id="description-text",
            style={"textAlign": "center", "marginTop": "10px", "fontSize": "16px"},
        ),
        dcc.Graph(id="indicator-graph"),
    ],
    style={"padding": "30px"},
)


# 🔁 콜백 함수
@app.callback(
    [Output("indicator-graph", "figure"), Output("description-text", "children")],
    [Input("indicator-dropdown", "value")],
)
def update_graph(selected_indicator):
    df = all_data[all_data["Series"] == selected_indicator]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Value"], mode="lines", name=selected_indicator)
    )
    fig.update_layout(
        title=selected_indicator,
        xaxis_title="Date",
        yaxis_title="Value",
        height=500,
    )
    description = indicator_descriptions.get(selected_indicator, "")
    return fig, description


# ▶ 실행
if __name__ == "__main__":
    app.run(debug=True)
