from commun_imports import *
from model_utils import embed_query, normalize_vector 
from keywords import load_dynamic_keywords 

config = load_config()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(config['index_name'])

SILENCE_DB_THRESHOLD = config['SILENCE_DB_THRESHOLD'] 
SILENCE_DURATION = config['SILENCE_DURATION']  
CHUNK = config['CHUNK']
FRAME_RATE = config['FRAME_RATE']

class VoiceAssistant:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=config['CHANNELS'],
            rate=config['FRAME_RATE'],
            input=True,
            frames_per_buffer=CHUNK
        )
        self.transcript = ""
        self.listening = True
        self.exit_program = False  
        self.silence_start = None
        
        self.dynamic_keywords = load_dynamic_keywords()  
        self.financial_keywords = config['financial_keywords']  
        self.combined_keywords = set(self.financial_keywords + self.dynamic_keywords)  

    async def process_audio(self):
        async with websockets.connect(config['DEEPGRAM_URL'], extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}) as ws:
            async def sender(ws):
                try:
                    while self.listening:
                        data = self.stream.read(CHUNK)
                        await ws.send(data)

                        # Silence Detection
                        rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16) ** 2))
                        db_level = 20 * np.log10(rms) if rms > 0 else 0
                        if db_level < SILENCE_DB_THRESHOLD:
                            if self.silence_start is None:
                                self.silence_start = time.time()  # Start silence timer
                            elif time.time() - self.silence_start >= SILENCE_DURATION:
                                print("Silence détecté, arrêt de l'écoute.")
                                self.listening = False
                                break  
                        else:
                            self.silence_start = None  

                except websockets.exceptions.ConnectionClosedOK:
                    pass
                except Exception as e:
                    print(f"Error in sender: {e}")

            async def receiver(ws):
                try:
                    async for msg in ws:
                        res = json.loads(msg)
                        if res.get("is_final"):
                            self.transcript += res["channel"]["alternatives"][0]["transcript"] + " "

                            if config['STOP_COMMAND'].strip().lower() in self.transcript.strip().lower():
                                self.exit_program = True
                                self.listening = False
                                break 

                except Exception as e:
                    print(f"Error in receiver: {e}")

            await asyncio.gather(sender(ws), receiver(ws))

    async def retrieve_relevant_info(self, query):
        try:
            if not query.strip(): 
                return ""
            query_vector = embed_query(query)
            normalized_query_vector = normalize_vector(query_vector)
            results = index.query(
                vector=normalized_query_vector,
                top_k=3,
                include_metadata=True
            )
            retrieved_context = " ".join(match['metadata']['text'] for match in results['matches'])
            return retrieved_context.strip()
        except Exception as e:
            print(f"Error retrieving relevant information: {e}")
            return ""

    def is_relevant_query(self, query):
        """
        This function checks if the query is relevant to the financial sector.
        You can expand the list of keywords based on your domain knowledge.
        """
        query_words = set(query.lower().split())
        return any(keyword in query_words for keyword in self.combined_keywords)

    def is_response_in_context(self, response, context):
        """
        This function checks if the LLM's response stays within the bounds of the provided context.
        It compares the words in the response with the context to make sure they're aligned.
        """
        
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        overlap = response_words.intersection(context_words)

        if len(overlap) / len(response_words) < 0.3:  
            return False

        return True

    async def get_ai_response(self, query, retrieved_context):
        if not query.strip():
            return "Désolé, votre question est vide ou non compréhensible."

        if not self.is_relevant_query(query):
            return "Je ne peux pas répondre à cette question, je traite uniquement des informations financières."

        if not retrieved_context:
            return "Désolé, aucune information pertinente n'a été trouvée dans la base de données."

        try:
            prompt = (f"Vous êtes un assistant financier. Vous devez strictement utiliser les informations présentes "
                      f"dans le dataset ci-dessous. Ne générez aucune information qui ne se trouve pas dans ces données. "
                      f"Si aucune information pertinente n'est disponible, indiquez que vous ne pouvez pas fournir d'information.\n"
                      f"Contexte pertinent:\n{retrieved_context}\n"
                      f"Question: {query}\n"
                      "Réponse:")

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt).text
            
            if not self.is_response_in_context(response, retrieved_context):
                return "Désolé, je ne peux pas fournir une réponse valide avec les informations disponibles."

            return response

        except Exception as e:
            print(f"Error in getting AI response: {e}")
            return "Désolé, je ne peux pas traiter votre demande pour le moment."

    async def run(self):
        while not self.exit_program: 
            print("En écoute...")
            self.transcript = ""
            self.listening = True
            self.silence_start = None  
            await self.process_audio()

            if self.exit_program:
                print("Sortie du programme...")
                sys.exit()  

            print(f"Vous avez dit: {self.transcript}")

            retrieved_context = await self.retrieve_relevant_info(self.transcript)
            ai_response = await self.get_ai_response(self.transcript, retrieved_context)
            print(f"Reponse de l'assistant: {ai_response}")
            text_to_speech(ai_response)

    def __del__(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
