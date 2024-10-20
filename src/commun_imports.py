import asyncio
import json
import os
import pyaudio
import websockets
from dotenv import load_dotenv
from text_to_speech import text_to_speech
import ffmpeg
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import numpy as np
import sys  
import time  

load_dotenv()

def load_config():
    with open('src/config.json', 'r') as file:
        return json.load(file)
