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
    print(f"📥 {indicators[series_id]} 데이터 로딩 중...")
    data = fred.get_series(series_id)
    df = data.reset_index()
    df.columns = ["Date", "Value"]
    df["Series"] = indicators[series_id]
    return df


# 🔁 모든 지표 통합
print("🚀 FRED 데이터 로딩 시작...")
all_data = pd.concat([fetch_data(sid) for sid in indicators.keys()])
all_data.dropna(inplace=True)
print(f"✅ 총 {len(all_data)} 개의 데이터 포인트 로딩 완료!")


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
def calculate_liquidity_score(df, target_year=None, target_month=None):
    score_dict = {}
    scaler = MinMaxScaler()

    for series_name in df["Series"].unique():
        df_sub = df[df["Series"] == series_name].sort_values("Date")

        if target_year:
            if target_month:
                # 특정 연도-월의 마지막 데이터 찾기
                df_period = df_sub[
                    (df_sub["Date"].dt.year == target_year)
                    & (df_sub["Date"].dt.month == target_month)
                ]
                if df_period.empty:
                    # 해당 월에 데이터가 없으면 해당 연도 해당 월 이전까지의 마지막 데이터
                    df_period = df_sub[
                        (df_sub["Date"].dt.year == target_year)
                        & (df_sub["Date"].dt.month <= target_month)
                    ]
                    if df_period.empty:
                        continue
            else:
                # 특정 연도의 마지막 데이터 찾기
                df_period = df_sub[df_sub["Date"].dt.year == target_year]
                if df_period.empty:
                    continue

            target_date = df_period["Date"].max()
            target_idx = df_sub[df_sub["Date"] == target_date].index[0]
            target_position = df_sub.index.get_loc(target_idx)

        values = df_sub["Value"].values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)

        if target_year:
            if target_position < len(scaled):
                latest = scaled[target_position][0]
            else:
                continue
        else:
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


# ✅ 현재 유동성 점수 계산
liquidity_score = calculate_liquidity_score(all_data)


# 📊 전체 기간 유동성 점수 히스토리 계산
def calculate_liquidity_score_history(df):
    """전체 기간에 대한 유동성 점수 히스토리를 계산 (월별 샘플링으로 최적화)"""
    score_history = []

    # 전체 데이터 범위를 기준으로 한 번에 정규화 (동일한 기준 사용)
    scaler_dict = {}
    for series_name in df["Series"].unique():
        series_data = df[df["Series"] == series_name].sort_values("Date")
        if len(series_data) > 0:
            values = series_data["Value"].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(values)  # 전체 범위로 스케일러 훈련
            scaler_dict[series_name] = scaler

    # 월별 마지막 날짜만 선택 (성능 최적화)
    print("📊 유동성 점수 히스토리 계산 중... (월별 샘플링)")
    # 월별 마지막 날짜 추출
    monthly_dates = (
        df.groupby([df["Date"].dt.year, df["Date"].dt.month])["Date"].max().tolist()
    )
    monthly_dates = sorted(monthly_dates)

    for i, date in enumerate(monthly_dates):
        if i % 12 == 0:  # 1년마다 진행상황 출력
            print(f"⏳ {date.year}년 처리 중...")

        # 해당 날짜까지의 데이터 필터링
        date_data = df[df["Date"] <= date]
        if len(date_data) == 0:
            continue

        score_dict = {}

        # 각 지표별로 해당 날짜의 점수 계산
        for series_name in df["Series"].unique():
            series_data = date_data[date_data["Series"] == series_name].sort_values(
                "Date"
            )
            if len(series_data) == 0 or series_name not in scaler_dict:
                continue

            # 해당 날짜의 최신 값 가져오기
            latest_value = series_data.iloc[-1]["Value"]

            # 전체 범위 기준으로 정규화된 값 계산
            scaled_value = scaler_dict[series_name].transform([[latest_value]])[0][0]

            # 역방향 지표 처리
            if series_name in [
                "Reverse Repo (RRP)",
                "Federal Funds Rate",
                "3-Month Treasury Rate",
                "Dollar Index (EUR/USD)",
            ]:
                scaled_value = 1 - scaled_value

            score_dict[series_name] = scaled_value

        # 가중 평균 계산
        if len(score_dict) > 0:
            weighted_score = sum(
                score_dict[series] * indicator_weights[series]
                for series in score_dict.keys()
                if series in indicator_weights
            )
            score_history.append(
                {"Date": date, "Score": round(weighted_score * 100, 1)}
            )

    print("✅ 유동성 점수 히스토리 계산 완료!")

    return pd.DataFrame(score_history)


# 점수 히스토리 데이터 생성
liquidity_score_history = calculate_liquidity_score_history(all_data)

# 📅 사용 가능한 연도 및 월 목록 생성
available_years = sorted(all_data["Date"].dt.year.unique(), reverse=True)
current_year = all_data["Date"].dt.year.max()
current_month = all_data[all_data["Date"].dt.year == current_year][
    "Date"
].dt.month.max()

# 월 목록 생성 (1월~12월)
months = [
    {"label": "1월", "value": 1},
    {"label": "2월", "value": 2},
    {"label": "3월", "value": 3},
    {"label": "4월", "value": 4},
    {"label": "5월", "value": 5},
    {"label": "6월", "value": 6},
    {"label": "7월", "value": 7},
    {"label": "8월", "value": 8},
    {"label": "9월", "value": 9},
    {"label": "10월", "value": 10},
    {"label": "11월", "value": 11},
    {"label": "12월", "value": 12},
]


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
        # 연도 및 월 선택 드롭다운
        html.Div(
            [
                html.Label(
                    "📅 기간 선택:",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "marginBottom": "10px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "연도:",
                                    style={"fontSize": "14px", "marginBottom": "5px"},
                                ),
                                dcc.Dropdown(
                                    id="year-dropdown",
                                    options=[
                                        {"label": f"{year}년", "value": year}
                                        for year in available_years
                                    ],
                                    value=current_year,
                                    style={"width": "150px"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "20px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "월:",
                                    style={"fontSize": "14px", "marginBottom": "5px"},
                                ),
                                dcc.Dropdown(
                                    id="month-dropdown",
                                    options=months,
                                    value=current_month,
                                    style={"width": "120px"},
                                ),
                            ],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "flex-end",
                    },
                ),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        # 동적 점수 표시
        html.H2(
            id="liquidity-score-display",
            style={"textAlign": "center", "color": "#1a73e8"},
        ),
        html.P(
            id="score-message",
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
        # 유동성 점수 히스토리 차트 섹션
        html.Hr(style={"margin": "40px 0"}),
        html.H3(
            "📈 유동성 점수 히스토리",
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        dcc.Graph(id="liquidity-score-chart"),
    ],
    style={"padding": "30px"},
)


# 🔁 콜백 함수 - 유동성 점수 업데이트
@app.callback(
    [
        Output("liquidity-score-display", "children"),
        Output("score-message", "children"),
    ],
    [Input("year-dropdown", "value"), Input("month-dropdown", "value")],
)
def update_liquidity_score(selected_year, selected_month):
    score = calculate_liquidity_score(all_data, selected_year, selected_month)
    score_text = f"💧 {selected_year}년 {selected_month}월 유동성 점수: {score} / 100"
    message = get_score_message(score)
    return score_text, message


# 🔁 콜백 함수 - 그래프 업데이트
@app.callback(
    [Output("indicator-graph", "figure"), Output("description-text", "children")],
    [
        Input("indicator-dropdown", "value"),
        Input("year-dropdown", "value"),
        Input("month-dropdown", "value"),
    ],
)
def update_graph(selected_indicator, selected_year, selected_month):
    df = all_data[all_data["Series"] == selected_indicator]

    # 선택된 연도-월까지의 데이터만 표시
    if selected_year and selected_month:
        df = df[
            (df["Date"].dt.year < selected_year)
            | (
                (df["Date"].dt.year == selected_year)
                & (df["Date"].dt.month <= selected_month)
            )
        ]
    elif selected_year:
        df = df[df["Date"].dt.year <= selected_year]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Value"], mode="lines", name=selected_indicator)
    )

    # 선택된 연도-월의 마지막 데이터 포인트 강조
    if selected_year and selected_month and not df.empty:
        period_data = df[
            (df["Date"].dt.year == selected_year)
            & (df["Date"].dt.month == selected_month)
        ]
        if period_data.empty:
            # 해당 월에 데이터가 없으면 해당 연도 해당 월 이전의 마지막 데이터
            period_data = df[
                (df["Date"].dt.year == selected_year)
                & (df["Date"].dt.month <= selected_month)
            ]

        if not period_data.empty:
            latest_point = period_data.iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[latest_point["Date"]],
                    y=[latest_point["Value"]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name=f"{selected_year}년 {selected_month}월 값",
                )
            )
    elif selected_year and not df.empty:
        year_data = df[df["Date"].dt.year == selected_year]
        if not year_data.empty:
            latest_point = year_data.iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[latest_point["Date"]],
                    y=[latest_point["Value"]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name=f"{selected_year}년 값",
                )
            )

    title_text = f"{selected_indicator}"
    if selected_year and selected_month:
        title_text += f" ({selected_year}년 {selected_month}월까지)"
    elif selected_year:
        title_text += f" ({selected_year}년까지)"

    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value",
        height=500,
    )
    description = indicator_descriptions.get(selected_indicator, "")
    return fig, description


# 🔁 콜백 함수 - 유동성 점수 히스토리 차트 업데이트
@app.callback(
    Output("liquidity-score-chart", "figure"),
    [Input("year-dropdown", "value"), Input("month-dropdown", "value")],
)
def update_liquidity_score_chart(selected_year, selected_month):
    fig = go.Figure()

    # 전체 히스토리 점수 라인 차트
    fig.add_trace(
        go.Scatter(
            x=liquidity_score_history["Date"],
            y=liquidity_score_history["Score"],
            mode="lines",
            name="유동성 점수",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # 선택된 시점 강조 표시
    if selected_year and selected_month:
        # 선택된 연도-월에 해당하는 데이터 찾기
        target_data = liquidity_score_history[
            (liquidity_score_history["Date"].dt.year == selected_year)
            & (liquidity_score_history["Date"].dt.month == selected_month)
        ]

        if target_data.empty:
            # 해당 월에 데이터가 없으면 해당 연도 해당 월 이전의 마지막 데이터
            target_data = liquidity_score_history[
                (liquidity_score_history["Date"].dt.year == selected_year)
                & (liquidity_score_history["Date"].dt.month <= selected_month)
            ]

        if not target_data.empty:
            latest_point = target_data.iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[latest_point["Date"]],
                    y=[latest_point["Score"]],
                    mode="markers",
                    name=f"{selected_year}년 {selected_month}월 점수",
                    marker=dict(size=12, color="red", symbol="diamond"),
                )
            )

    # 점수 구간별 색상 구역 표시
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="green",
        opacity=0.5,
        annotation_text="매우 풍부 (80+)",
    )
    fig.add_hline(
        y=60,
        line_dash="dash",
        line_color="orange",
        opacity=0.5,
        annotation_text="충분 (60+)",
    )
    fig.add_hline(
        y=40,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text="주의 (40+)",
    )

    # 배경 색상 구역 추가
    fig.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=40, y1=60, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.1, line_width=0)

    fig.update_layout(
        title="🌊 시간에 따른 유동성 점수 변화",
        xaxis_title="날짜",
        yaxis_title="유동성 점수",
        height=500,
        hovermode="x unified",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ▶ 실행
if __name__ == "__main__":
    print("🌊 유동성 대시보드 서버 시작...")
    print("🔗 브라우저에서 http://127.0.0.1:8050 접속하세요!")
    app.run(debug=False, host="127.0.0.1", port=8050)
