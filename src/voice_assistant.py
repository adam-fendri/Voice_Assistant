import asyncio
import json
import os
import pyaudio
import websockets
from openai import AsyncOpenAI
from dotenv import load_dotenv
from text_to_speech import text_to_speech
import ffmpeg

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = AsyncOpenAI(api_key=OPENAI_API_KEY)


CHANNELS = 1
FRAME_RATE = 16000
CHUNK = 8000
RECORDING_FILE = "recording.wav"
CONVERTED_FILE = "converted.mp3"


DEEPGRAM_URL = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={FRAME_RATE}&channels={CHANNELS}&model=nova-2"

STOP_COMMAND = "stop"

class VoiceAssistant:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=FRAME_RATE, input=True, frames_per_buffer=CHUNK)
        self.transcript = ""
        self.listening = True

    async def process_audio(self):
        async with websockets.connect(DEEPGRAM_URL, extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}) as ws:
            async def sender(ws):
                try:
                    while self.listening:
                        data = self.stream.read(CHUNK)
                        await ws.send(data)
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                except Exception as e:
                    print(f"Error in sender: {e}")

            async def receiver(ws):
                nonlocal self
                try:
                    async for msg in ws:
                        res = json.loads(msg)
                        if res.get("is_final"):
                            self.transcript += res["channel"]["alternatives"][0]["transcript"] + " "
                            if STOP_COMMAND in self.transcript.lower():
                                self.listening = False
                                return
                            
                except Exception as e:
                    print(f"Error in receiver: {e}")

            await asyncio.gather(sender(ws), receiver(ws))

    async def get_ai_response(self):
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond naturally as if you're a human."},
                    {"role": "user", "content": self.transcript}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in getting AI response: {e}")
            return "Sorry, I couldn't process that request."

    def convert_audio(self):
        try:
            ffmpeg.input(RECORDING_FILE).output(CONVERTED_FILE).run()
            print(f"Audio converted to {CONVERTED_FILE}")
        except Exception as e:
            print(f"Error in audio conversion: {e}")

    async def run(self):
        while self.listening:
            print("Listening...")
            self.transcript = ""
            await self.process_audio()
            if not self.listening:
                print("Stopped listening.")
                #self.convert_audio()  # Convert audio after stopping
                #break
            print(f"You said: {self.transcript}")
            
            ai_response = await self.get_ai_response()
            print(f"AI response: {ai_response}")
            
            text_to_speech(ai_response)  # Call the function from text_to_speech.py
            self.listening = True
    def __del__(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()


