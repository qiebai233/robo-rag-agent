import streamlit as st

from agent.react_agent import ReactAgent

st.title("智能扫地机器人客服")
st.divider()


def capture(generator, cache_list):
    """边往外吐数据，边把吐出来的内容缓存下来，方便存进会话历史"""
    for chunk in generator:
        cache_list.append(chunk)
        yield chunk


if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

#输入提示词
prompt = st.chat_input("请输入你的问题，例如：扫地机器人充不进电怎么办")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        with st.chat_message("assistant"):
            answer = st.write_stream(capture(st.session_state["agent"].execute_stream(prompt), response_messages))

    st.session_state["messages"].append({"role": "assistant", "content": answer})
