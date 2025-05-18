import streamlit as st
import asyncio
import websockets
import json

async def connect_to_websocket(app_id: str):
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    try:
        async with websockets.connect(uri) as websocket:
            st.success("[open] Connection established")

            send_message = json.dumps({"ping": 1})
            await websocket.send(send_message)
            st.info("Ping sent to server")
            st.code(send_message, language="json")

            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            st.success("[message] Data received from server:")
            st.json(json.loads(response))

    except asyncio.TimeoutError:
        st.error("Timeout: No response from server.")
    except websockets.ConnectionClosedError as e:
        if e.code == 1000:
            st.warning(f"[close] Clean connection closed: code={e.code}, reason={e.reason}")
        else:
            st.error("[close] Connection died")
    except Exception as e:
        st.error(f"[error] {str(e)}")

def run_async(func, *args):
    return asyncio.new_event_loop().run_until_complete(func(*args))

# --- Streamlit UI ---
st.title("Deriv WebSocket Ping Tester")

app_id = st.text_input("Enter your Deriv App ID", "")
run = st.button("Connect and Send Ping")

if run:
    if app_id:
        st.write("Connecting...")
        run_async(connect_to_websocket, app_id)
    else:
        st.warning("Please enter a valid App ID.")
