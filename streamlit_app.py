import asyncio
import websockets
import json

async def connect_to_websocket():
    app_id = app_id  # Replace with your app_id
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"  # WebSocket URI with the app_id

    try:
        # Establish a connection to the WebSocket server
        async with websockets.connect(uri) as websocket:
            print("[open] Connection established")  # Connection opened
            print("Sending to server")

            # Prepare the message to send (ping message in JSON format)
            send_message = json.dumps({"ping": 1})
            await websocket.send(send_message)  # Send the ping message to the server

            # Wait for a response from the server
            response = await websocket.recv()
            print(f"[message] Data received from server: {response}")  # Log the server's response

    except websockets.ConnectionClosedError as e:
        # Handle the scenario where the connection is closed
        if e.code == 1000:
            print(f"[close] Connection closed cleanly, code={e.code} reason={e.reason}")  # Clean close
        else:
            print("[close] Connection died")  # Abrupt close, likely due to network or server issues

    except Exception as e:
        # Handle any other exceptions that may occur
        print(f"[error] {str(e)}")  # Log any errors that occur

# Run the WebSocket client
# asyncio.get_event_loop().run_until_complete() starts the coroutine connect_to_websocket()
asyncio.get_event_loop().run_until_complete(connect_to_websocket())
