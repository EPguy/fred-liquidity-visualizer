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

# ✅ 지표 목록 및 매핑 - 개선된 유동성 지표들
indicators = {
    "RRPONTSYD": "Reverse Repo (RRP)",
    "M2SL": "M2 Money Supply",
    "FEDFUNDS": "Federal Funds Rate",
    "BOGZ1FL193020005Q": "Bank Reserves",  # 은행 준비금 - 핵심 유동성 지표
    "DGS3MO": "3-Month Treasury Rate",  # 3개월 국채금리 - 단기 유동성 반영
    "DEXUSEU": "Dollar Index (EUR/USD)",  # 달러 강세 - 글로벌 유동성 영향
    # 참고용 지표들 (점수 계산에는 미포함)
    "WDTGAL": "TGA (Treasury General Account)",
    "BUSLOANS": "C&I Loans",
    "BAMLH0A0HYM2": "High-Yield Spread",
}

# 📝 한글 설명 - 개선된 지표들
indicator_descriptions = {
    "Reverse Repo (RRP)": "역레포는 연준이 초과 유동성을 흡수할 때 사용하는 단기 자금 운용 수단입니다.",
    "M2 Money Supply": "M2는 광의 통화량으로, 현금, 요구불 예금, 저축성 예금 등을 포함합니다.",
    "Federal Funds Rate": "연방기금금리는 미국 은행 간 초단기 금리로, 연준의 통화 정책 방향을 반영합니다.",
    "Bank Reserves": "은행 준비금은 은행들이 연준에 보유한 초과 자금으로, 시장 유동성의 직접적인 지표입니다. 📈 높을수록: 은행들이 여유 자금이 많아 유동성 풍부, 📉 낮을수록: 은행 시스템 내 유동성 부족을 의미합니다.",
    "3-Month Treasury Rate": "3개월 국채금리는 단기 유동성 상황을 잘 반영하며, 연준 정책과 시장 기대를 나타냅니다. 📈 높을수록: 단기 자금 조달 비용 상승으로 유동성 긴축, 📉 낮을수록: 단기 유동성 풍부를 의미합니다.",
    "Dollar Index (EUR/USD)": "달러 대비 유로 환율로, 달러 강세는 글로벌 유동성 긴축 효과를 가져옵니다. 📈 높을수록: 달러 약세로 글로벌 유동성 완화, 📉 낮을수록: 달러 강세로 글로벌 유동성 긴축 효과를 의미합니다.",
    # 참고용 지표들 (점수 계산에는 미포함)
    "TGA (Treasury General Account)": "재무부 일반 계정은 미 재무부가 연준에 보유한 예산 잔액을 나타냅니다. 📈 높을수록: 정부 자금이 연준에 보관되어 유동성 흡수, 📉 낮을수록: 정부 지출로 시장에 유동성 공급을 의미합니다.",
    "C&I Loans": "기업 대출(C&I Loans)은 은행이 기업에 빌려준 자금으로, 기업의 신용 수요와 경기 활력을 보여줍니다. 📈 높을수록: 기업 대출 수요 증가로 경기 활성화, 📉 낮을수록: 신용 수요 감소로 경기 둔화를 의미합니다.",
    "High-Yield Spread": "하이일드 스프레드는 고위험 회사채와 국채의 금리 차이를 말하며, 금융시장의 스트레스 수준을 반영합니다. 📈 높을수록: 시장 불안으로 위험 회피 심화, 📉 낮을수록: 시장 안정으로 위험 선호 증가를 의미합니다.",
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


# ⚖️ 지표별 가중치 설정 - 개선된 배분
indicator_weights = {
    "Reverse Repo (RRP)": 0.30,  # 최고 가중치 - 연준의 직접적 유동성 흡수 도구
    "Bank Reserves": 0.25,  # 높은 가중치 - 은행 시스템 유동성의 핵심 지표
    "M2 Money Supply": 0.20,  # 높은 가중치 - 전체 통화량 기반
    "Federal Funds Rate": 0.15,  # 중간 가중치 - 정책금리 (다른 지표들이 이미 반영)
    "3-Month Treasury Rate": 0.07,  # 낮은 가중치 - 단기 유동성 보완 지표
    "Dollar Index (EUR/USD)": 0.03,  # 최소 가중치 - 글로벌 유동성 영향 (보조 지표)
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
            "Reverse Repo (RRP)",  # 높을수록 유동성 흡수
            "Federal Funds Rate",  # 높을수록 긴축 정책
            "3-Month Treasury Rate",  # 높을수록 단기 유동성 긴축
            "Dollar Index (EUR/USD)",  # 달러 강세는 글로벌 유동성 긴축
        ]:
            latest = 1 - latest

        score_dict[series_name] = latest

    # 가중 평균 계산 (가중치가 있는 지표들만)
    weighted_score = sum(
        score_dict[series] * indicator_weights[series]
        for series in score_dict.keys()
        if series in indicator_weights
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
