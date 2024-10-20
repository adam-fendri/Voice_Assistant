import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[3].id)
engine.setProperty('rate', 170) 


def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


