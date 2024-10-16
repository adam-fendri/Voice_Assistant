import asyncio
from voice_assistant import VoiceAssistant

if __name__ == "__main__":
    assistant = VoiceAssistant()
    asyncio.run(assistant.run())