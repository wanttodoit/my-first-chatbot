import os
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI

# -----------------------
# 0. 기본 설정
# -----------------------
load_dotenv()

st.set_page_config(
    page_title="나의 첫 AI 챗봇",
    page_icon="🤖",
    layout="wide",
)

# -----------------------
# 1. Azure OpenAI 클라이언트
# -----------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OAI_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT"),
)

# -----------------------
# 2. Session State 초기화
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 친절하게 대답하는 한국어 AI 비서입니다."

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

# -----------------------
# 3. 사이드바 (꾸미기 + 설정)
# -----------------------
with st.sidebar:
    st.header("⚙️ 설정")

    # 시스템 프롬프트
    st.markdown("**🧠 AI 역할 설정**")
    system_prompt_input = st.text_area(
        "시스템 메시지",
        value=st.session_state.system_prompt,
        height=100,
        label_visibility="collapsed",
    )
    st.session_state.system_prompt = system_prompt_input

    # 온도 슬라이더
    st.markdown("**🔥 창의성(temperature)**")
    temp = st.slider("창의성", 0.0, 1.5, st.session_state.temperature, 0.1)
    st.session_state.temperature = temp

    st.markdown("---")

    # 대화 초기화 버튼
    if st.button("🧹 대화 초기화"):
        st.session_state.messages = []
        st.experimental_rerun()

# -----------------------
# 4. 메인 화면 헤더
# -----------------------
st.title("🤖 나의 첫 AI 챗봇")
st.caption("Azure OpenAI + Streamlit 데모")

# -----------------------
# 5. 이전 대화 출력
# -----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------
# 6. 사용자 입력 받기
# -----------------------
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    # 1) 사용자 메시지 화면에 표시 & 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2) AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            # system 메시지 + 지금까지 대화 모아서 보냄
            messages_for_api = [
                {"role": "system", "content": st.session_state.system_prompt},
                *st.session_state.messages,
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 배포 이름
                messages=messages_for_api,
                temperature=st.session_state.temperature,
            )
            assistant_reply = response.choices[0].message.content

            st.markdown(assistant_reply)

    # 3) AI 응답 저장
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
