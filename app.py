import os
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI

# -----------------------
# 0. 기본 설정
# -----------------------
load_dotenv()

st.set_page_config(
    page_title="UAM Paper Reference",
    page_icon="📚",
    layout="wide",
)

# ===== 공통 설정 함수: env → secrets 순서로 읽기 =====
def get_config(key: str, default: str | None = None) -> str | None:
    """
    1) os.getenv 에서 먼저 찾고
    2) 없으면 st.secrets[key] 를 시도 (로컬에는 없을 수 있으니 try/except)
    """
    v = os.getenv(key)
    if v:
        return v
    try:
        return st.secrets[key]
    except Exception:
        return default

# -----------------------
# 1. Azure OpenAI 클라이언트
# -----------------------
endpoint = get_config("AZURE_OAI_ENDPOINT")
api_key = get_config("AZURE_OAI_KEY")
deployment = get_config("AZURE_OAI_DEPLOYMENT", "gpt-4o-mini")

if not endpoint or not api_key:
    st.error(
        "⚠️ Azure OpenAI 설정을 찾을 수 없습니다.\n\n"
        "다음 값이 설정되어 있는지 확인해 주세요.\n"
        "- AZURE_OAI_ENDPOINT\n"
        "- AZURE_OAI_KEY\n"
        "- (선택) AZURE_OAI_DEPLOYMENT"
    )
    st.stop()

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-05-01-preview",
    azure_endpoint=endpoint,
)

MODEL_NAME = deployment  # 아래에서 사용할 모델 이름

# ===== CSS (하늘색 라이트 테마) =====
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #e0f2fe 0%, #f9fafb 40%, #ffffff 100%);
        color: #0f172a;
    }

    .uam-hero {
        padding: 1.5rem 1.8rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #0ea5e9, #3b82f6);
        border: 1px solid #bae6fd;
        box-shadow: 0 18px 45px rgba(15,23,42,0.25);
        color: #e5f3ff;
    }

    .uam-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.25);
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #e5f3ff;
    }

    .uam-badge span {
        font-size: 0.9rem;
    }

    .uam-subtext {
        color: #dbeafe;
        font-size: 0.88rem;
    }

    .uam-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(191,219,254,0.9);
        font-size: 0.78rem;
        color: #e0f2fe;
        margin-right: 0.35rem;
        background: rgba(15,23,42,0.25);
    }

    .uam-card {
        padding: 0.9rem 0.9rem;
        border-radius: 0.9rem;
        border: 1px solid #bfdbfe;
        background: #f0f9ff;
        font-size: 0.85rem;
        box-shadow: 0 10px 25px rgba(148,163,184,0.35);
    }
    .uam-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 0.98rem;
        color: #0f172a;
    }
    .uam-card p {
        margin: 0 0 0.35rem 0;
        font-size: 0.8rem;
        color: #1f2937;
    }
    .uam-tag {
        display: inline-block;
        font-size: 0.72rem;
        padding: 0.12rem 0.45rem;
        border-radius: 999px;
        background: #38bdf8;
        color: #f9fafb;
        margin-top: 0.2rem;
    }

    .uam-footer {
        margin-top: 1.2rem;
        padding-top: 0.6rem;
        border-top: 1px solid #e5e7eb;
        font-size: 0.75rem;
        color: #64748b;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# 2. Session State 초기화
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    # 기본 역할: 논문 추천 + 줄거리 + DOI
    st.session_state.system_prompt = (
        "너는 Urban Air Mobility(UAM), eVTOL, vertiport, 도심항공교통과 관련된 "
        "학술 논문을 찾아주고 정리해 주는 레퍼런스 도우미야. "
        "사용자가 한국어로 관심 주제(예: 입지선정, vertiport 용량 분석, 스케줄링, 소음/수용성, "
        "수요예측, 안전/위험도 등)를 설명하면, 가능한 범위 안에서 실제로 존재하는 영어 논문을 "
        "위주로 추천해 줘. 각 논문마다 (1) 저자, 연도, 제목, 학회/저널, "
        "(2) 한 줄 요약, (3) 핵심 키워드, (4) 왜 사용자의 주제에 도움이 되는지, "
        "(5) 논문의 줄거리(연구 배경–문제 설정–방법–주요 결과–시사점)를 3~5문장 정도로 "
        "자연스럽게 한국어로 정리해 줘. "
        "(6) DOI를 알고 있을 때만 정확한 값을 적고, 확실하지 않으면 반드시 'DOI: unknown'이라고 써 줘. "
        "절대로 추측해서 DOI를 만들어내지 마."
    )

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5

if "system_prompt_draft" not in st.session_state:
    st.session_state.system_prompt_draft = st.session_state.system_prompt

# -----------------------
# 3. 사이드바 - 프롬프트/모드
# -----------------------
with st.sidebar:
    st.header("📚 UAM 논문 탐색 패널")

    st.markdown("**🎛 AI 기본 역할 프롬프트**")
    st.text_area(
        "시스템 프롬프트",
        key="system_prompt_draft",
        height=160,
        label_visibility="collapsed",
    )

    if st.button("등록", use_container_width=True):
        st.session_state.system_prompt = st.session_state.system_prompt_draft
        st.success("역할 프롬프트가 적용되었습니다.")

    st.markdown("**🔥 다양성 조절 (temperature)**")
    temp = st.slider("창의성", 0.0, 1.5, float(st.session_state.temperature), 0.1)
    st.session_state.temperature = temp

    st.markdown("---")
    st.markdown("**🔎 검색 모드**")
    mode = st.radio(
        "검색 모드",
        ["기본 검색", "리뷰/Survey 위주", "방법론/모델 위주", "케이스 스터디 위주"],
        label_visibility="collapsed",
    )

    mode_suffix = ""
    if mode == "리뷰/Survey 위주":
        mode_suffix = (
            " 사용자가 제시한 주제와 관련된 review paper, survey paper, tutorial 논문을 "
            "우선적으로 추천해 줘. 각 리뷰 논문이 커버하는 범위를 설명해 줘."
        )
    elif mode == "방법론/모델 위주":
        mode_suffix = (
            " 통계/최적화/시뮬레이션/머신러닝 등 방법론 측면에서 의미 있는 논문을 "
            "우선적으로 추천해 줘. 각 논문이 사용한 모델·알고리즘·시뮬레이션 구조를 짧게 정리해 줘."
        )
    elif mode == "케이스 스터디 위주":
        mode_suffix = (
            " 특정 도시, 국가, 실증 프로젝트(UAM 시범도시, 시험 노선 등)를 다룬 케이스 스터디 "
            "논문을 우선적으로 추천해 줘. 어느 지역/도시를 대상으로 했는지도 함께 언급해 줘."
        )

    effective_system_prompt = st.session_state.system_prompt + mode_suffix

    st.markdown("---")
    if st.button("🧹 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.experimental_rerun()

# -----------------------
# 4. 메인 헤더
# -----------------------
st.markdown(
    """
    <div class="uam-hero">
        <div class="uam-badge">
            <span>📚</span> UAM · PAPER REFERENCE
        </div>
        <h1 style="margin: 0.6rem 0 0.3rem 0; font-size: 1.6rem;">
            UAM 연구를 위한 논문 레퍼런스 허브
        </h1>
        <p class="uam-subtext">
            "vertiport 입지선정", "버티포트 수용량 분석", "UAM 소음에 따른 수용성"처럼<br>
            궁금한 주제를 한국어로 적어주면, 관련 UAM 논문들을 찾아서 정리해 줍니다.
        </p>
        <div style="margin-top: 0.4rem;">
            <span class="uam-pill">Vertiport Siting</span>
            <span class="uam-pill">Capacity & Scheduling</span>
            <span class="uam-pill">Public Acceptance · Noise</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -----------------------
# 5. 프리셋 카드
# -----------------------
preset_prompt = None

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="uam-card">
          <h4>📍 Vertiport 입지선정 논문</h4>
          <p>도시 내 vertiport 후보지 선정, 다기준 의사결정(MCDM), GIS 기반 입지분석 관련 논문을 찾을 때.</p>
          <div class="uam-tag">Siting · Location</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("예시 불러오기", key="preset1"):
        preset_prompt = (
            "대도시 내 UAM vertiport 입지선정과 관련된 학술 논문을 추천해 줘. "
            "특히 GIS 분석, 다기준 의사결정(MCDM), AHP/ANP, 최적화 모델을 사용한 연구 위주로 알려줘."
        )

with col2:
    st.markdown(
        """
        <div class="uam-card">
          <h4>🛬 용량 · 스케줄링 논문</h4>
          <p>버티포트 수용량, 지상 처리 프로세스, arrival–departure scheduling 관련 논문을 찾을 때.</p>
          <div class="uam-tag">Capacity · Operations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("예시 불러오기", key="preset2"):
        preset_prompt = (
            "UAM vertiport 용량 분석과 arrival/departure 스케줄링 관련 논문을 추천해 줘. "
            "시뮬레이션(DES, 에이전트 기반, microsimulation)이나 최적화 모델을 사용한 연구 위주로 알려줘."
        )

with col3:
    st.markdown(
        """
        <div class="uam-card">
          <h4>👥 수용성 · 소음 · 안전 인식</h4>
          <p>UAM 이용자의 수용성, 소음 인식, 안전/위험 인식, 사회적 수용성(social acceptance)을 다룬 연구를 찾을 때.</p>
          <div class="uam-tag">Public Acceptance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("예시 불러오기", key="preset3"):
        preset_prompt = (
            "UAM 서비스에 대한 대중 수용성, 소음에 대한 인식, 안전/위험 인식, 사회적 수용성을 분석한 논문들을 추천해 줘. "
            "설문 기반 연구나 stated preference/choice experiment를 사용한 논문이 있으면 특히 알려줘."
        )

st.write("")

# -----------------------
# 6. 기존 대화 출력
# -----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------
# 7. 사용자 입력 + 프리셋
# -----------------------
user_input = st.chat_input("찾고 싶은 UAM 논문 주제를 한국어로 편하게 써 주세요.")

if preset_prompt and not user_input:
    prompt = preset_prompt
elif user_input:
    prompt = user_input
else:
    prompt = None

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("관련 UAM 논문들을 정리하는 중입니다..."):
            messages_for_api = [
                {"role": "system", "content": effective_system_prompt},
                *st.session_state.messages,
            ]

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_for_api,
                temperature=float(st.session_state.temperature),
            )
            assistant_reply = response.choices[0].message.content
            st.markdown(assistant_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

# -----------------------
# 8. 푸터
# -----------------------
st.markdown(
    """
    <div class="uam-footer">
      UAM Paper Reference · Built by Jae-Kyun
    </div>
    """,
    unsafe_allow_html=True,
)
