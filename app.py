from gtts import gTTS
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import random
import chromadb
import gradio as gr
import tempfile
from collections import deque
import re
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Initialize conversation memory (last 40 exchanges)
conversation_memory = deque(maxlen=40)

# Initialize models with error handling
try:
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    print("✓ Speech-to-text model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load speech-to-text model: {e}")
    stt_pipe = None

try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("✓ Sentiment analyzer loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load sentiment analyzer: {e}")
    sentiment_analyzer = None

def initialize_llm():
    try:
        llm = ChatGroq(
            temperature=0.8,
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        print("✓ LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return None

def create_vector_db():
    try:
        loader = DirectoryLoader(r"Training_doc", glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        print(f"✓ Generated {len(texts)} text chunks.")

        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
        vector_db.persist()
        return vector_db
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        return None

def get_conversation_context():
    if not conversation_memory:
        return "This is our first interaction - I'm excited to get to know you!"

    context_parts = ["Recent conversation highlights:"]
    for i, (user_msg, bot_response) in enumerate(list(conversation_memory)[-5:]):  # Last 5 exchanges
        context_parts.append(f"You mentioned: {user_msg[:100]}...")
        context_parts.append(f"I responded about: {bot_response[:100]}...")

    return "\n".join(context_parts)

def add_to_memory(user_input, bot_response):
    conversation_memory.append((user_input, bot_response))

def analyze_sentiment_enhanced(text):
    """Enhanced sentiment analysis with emotional nuance detection"""
    if not sentiment_analyzer:
        return {'label': 'NEUTRAL', 'score': 0.5, 'emotion': 'calm'}
    
    try:
        sentiment = sentiment_analyzer(text)[0]
        
        # Detect specific emotions based on keywords
        emotion_keywords = {
            'anxious': ['anxious', 'worried', 'nervous', 'panic', 'stress', 'overwhelmed'],
            'sad': ['sad', 'depressed', 'down', 'hopeless', 'lonely', 'empty'],
            'angry': ['angry', 'frustrated', 'mad', 'annoyed', 'furious', 'irritated'],
            'excited': ['excited', 'happy', 'thrilled', 'amazing', 'wonderful', 'fantastic'],
            'tired': ['tired', 'exhausted', 'drained', 'weary', 'burnt out'],
            'confused': ['confused', 'lost', 'unclear', 'don\'t understand', 'mixed up']
        }
        
        detected_emotion = 'neutral'
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                detected_emotion = emotion
                break
        
        return {
            'label': sentiment['label'],
            'score': sentiment['score'],
            'emotion': detected_emotion,
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {'label': 'NEUTRAL', 'score': 0.5, 'emotion': 'neutral'}

def audio_to_text(audio_path):
    """Convert audio to text with better error handling"""
    if audio_path is None or not stt_pipe:
        return None
    
    try:
        result = stt_pipe(audio_path)
        text = result["text"] if isinstance(result, dict) else str(result)
        print(f"🎤 Transcribed: {text}")
        return text.strip()
    except Exception as e:
        print(f"❌ Audio transcription error: {e}")
        return None

def text_to_audio(text):
    """Convert text to audio with better error handling"""
    if not text or len(text.strip()) == 0:
        return None
    
    try:
        # Clean text for TTS
        clean_text = re.sub(r'[^\w\s.,!?-]', '', text)
        clean_text = re.sub(r'\*\*.*?\*\*', '', clean_text)  # Remove markdown bold
        clean_text = re.sub(r'[•▪▫]', '', clean_text)  # Remove bullet points
        clean_text = clean_text.replace('\n', '. ')  # Replace newlines with periods
        
        if len(clean_text.strip()) == 0:
            return None
            
        # Limit text length for TTS
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        print(f"🔊 Audio generated: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None

def is_greeting(text):
    """Enhanced greeting detection"""
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 
        'how are you', 'what\'s up', 'howdy', 'greetings', 'yo'
    ]
    text_lower = text.lower().strip()
    return any(greeting in text_lower for greeting in greetings) and len(text.split()) <= 6

def generate_natural_response(user_input, sentiment_info, context):
    """Generate more natural, sentiment-aware responses"""
    
    # Emotion-based response starters
    emotion_responses = {
        'anxious': [
            "I can hear the worry in your words, and that's completely understandable.",
            "It sounds like you're carrying some heavy feelings right now.",
            "Those anxious thoughts can feel so overwhelming, can't they?"
        ],
        'sad': [
            "I'm really sorry you're going through such a tough time.",
            "It takes courage to share when you're feeling this way.",
            "Your feelings are valid, and I'm here with you through this."
        ],
        'angry': [
            "I can feel the frustration in what you're saying.",
            "It sounds like something really got under your skin.",
            "That anger makes total sense given what you're dealing with."
        ],
        'excited': [
            "I love hearing the excitement in your voice!",
            "That energy is contagious - tell me more!",
            "It sounds like something wonderful happened!"
        ],
        'tired': [
            "It sounds like you're running on empty right now.",
            "Being exhausted like that is really draining.",
            "Sometimes we all need to acknowledge when we're burnt out."
        ],
        'confused': [
            "It's okay to feel uncertain - life can be pretty confusing.",
            "Let's try to untangle this together, one piece at a time.",
            "Confusion often means we're processing something important."
        ]
    }
    
    # Select appropriate response starter based on emotion
    emotion = sentiment_info.get('emotion', 'neutral')
    if emotion in emotion_responses:
        starter_options = emotion_responses[emotion]
        return random.choice(starter_options)
    
    # Default natural starters
    default_starters = [
        "I hear you.",
        "That resonates with me.",
        "I'm glad you shared that.",
        "Thanks for being open with me.",
        "I appreciate you trusting me with this."
    ]
    
    return random.choice(default_starters)

def should_include_practical_tips(user_input, sentiment_info):
    """Determine if practical tips would be helpful"""
    help_seeking = any(phrase in user_input.lower() for phrase in [
        'what should i do', 'help me', 'any advice', 'suggestions', 'tips', 'how do i'
    ])
    
    negative_emotion = sentiment_info.get('emotion') in ['anxious', 'sad', 'tired', 'confused']
    
    return help_seeking or (negative_emotion and sentiment_info.get('score', 0) > 0.7)

def generate_contextual_tips(user_input, sentiment_info):
    """Generate tips based on specific context and emotion"""
    emotion = sentiment_info.get('emotion', 'neutral')
    
    tips_by_emotion = {
        'anxious': [
            "Try the 4-7-8 breathing: breathe in for 4, hold for 7, out for 8",
            "Ground yourself with 5 things you can see, 4 you can touch, 3 you can hear",
            "Remember: this feeling is temporary, like weather passing through"
        ],
        'sad': [
            "Be gentle with yourself - you'd comfort a friend this way",
            "Small steps count: even getting dressed or having tea is an achievement",
            "Consider reaching out to someone you trust, even just to say hello"
        ],
        'tired': [
            "Rest isn't lazy - it's necessary maintenance for your wellbeing",
            "Try a 10-minute walk outside, fresh air can be surprisingly refreshing",
            "Maybe tonight, prioritize sleep over productivity"
        ],
        'angry': [
            "Take 10 deep breaths before responding to whatever triggered this",
            "Physical movement can help release that angry energy - try stretching",
            "Ask yourself: 'Will this matter in 5 years?' for perspective"
        ],
        'confused': [
            "Write down your thoughts - sometimes seeing them helps clarify",
            "It's okay not to have all the answers right now",
            "Talk it through with someone you trust, or even out loud to yourself"
        ]
    }
    
    if emotion in tips_by_emotion:
        return random.sample(tips_by_emotion[emotion], min(2, len(tips_by_emotion[emotion])))
    
    return []

greeting_responses = [
    "Hey there! How are you feeling today?",
    "Hi! I'm really glad you're here. What's on your mind?",
    "Hello! It's nice to connect with you. How can I support you today?",
    "Hey! Thanks for stopping by. What would you like to talk about?",
    "Hi there! I'm here and ready to listen. What's happening in your world?"
]

def process_user_input(user_input, chat_history, is_voice=False):
    """Enhanced input processing with sentiment awareness"""
    if not user_input or not user_input.strip():
        return chat_history, None

    # Analyze sentiment first
    sentiment_info = analyze_sentiment_enhanced(user_input)
    print(f"💭 Detected emotion: {sentiment_info['emotion']}, sentiment: {sentiment_info['label']}")

    # Handle greetings
    if is_greeting(user_input):
        response = random.choice(greeting_responses)
        add_to_memory(user_input, response)
        audio_path = text_to_audio(response)
        display_input = f"🎤 {user_input}" if is_voice else user_input
        chat_history.append((display_input, response))
        return chat_history, audio_path

    # Get conversation context
    context = get_conversation_context()
    
    # Generate natural response starter
    natural_starter = generate_natural_response(user_input, sentiment_info, context)

    # Create enhanced query for the LLM
    sentiment_context = f"User's emotional state: {sentiment_info['emotion']} ({sentiment_info['label']})"
    
    enhanced_query = f"""
    {sentiment_context}
    Previous conversation context:
    {context}
    
    Current message: {user_input}
    
    Please respond as a warm, empathetic friend who truly understands emotions. Start your response naturally (don't use phrases like "I understand" or "Thank you for sharing"). Be conversational, genuine, and emotionally intelligent. Consider the user's emotional state and conversation history.
    """

    try:
        if qa_chain:
            response = qa_chain({"query": enhanced_query})
            ai_response = response.get("result", "I'm here to listen, tell me more.")
        else:
            # Fallback response
            ai_response = f"{natural_starter} I'm experiencing some technical difficulties but I'm still here to listen and support you."

        # Combine natural starter with AI response if needed
        if not ai_response.startswith(natural_starter.split('.')[0]):
            final_response = f"{natural_starter} {ai_response}"
        else:
            final_response = ai_response

        # Add practical tips if appropriate
        if should_include_practical_tips(user_input, sentiment_info):
            tips = generate_contextual_tips(user_input, sentiment_info)
            if tips:
                final_response += "\n\n💡 **A couple of things that might help:**\n"
                for tip in tips:
                    final_response += f"• {tip}\n"

        # Crisis intervention check
        crisis_keywords = [
            'suicide', 'self-harm', 'kill myself', 'cutting', 'hurt myself', 'end my life',
            'no reason to live', 'want to die', 'overdose', 'hang myself'
        ]

        if any(keyword in user_input.lower() for keyword in crisis_keywords):
            crisis_response = """

🆘 **I'm very concerned about what you're sharing. Your life has immense value.**

**Please reach out immediately:**
• **Crisis Text Line**: Text HOME to 741741
• **National Suicide Prevention Lifeline**: 988
• **India**: Vandrevala Foundation - 1860 266 2345
• **Emergency**: 911/112

You showed incredible strength by reaching out. These feelings are temporary, even when they feel endless. Professional help can make a real difference. Please connect with these resources - you're not alone. 💙"""
            
            final_response += crisis_response

        # Store in memory and generate audio
        add_to_memory(user_input, final_response)
        audio_path = text_to_audio(final_response)
        
        display_input = f"🎤 {user_input}" if is_voice else user_input
        chat_history.append((display_input, final_response))
        
        return chat_history, audio_path

    except Exception as e:
        error_response = f"{natural_starter} I'm experiencing some technical difficulties, but I'm still here with you. Can you tell me a bit more about what's on your mind?"
        add_to_memory(user_input, error_response)
        display_input = f"🎤 {user_input}" if is_voice else user_input
        chat_history.append((display_input, error_response))
        return chat_history, None

def gradio_chat(user_input, chat_history):
    """Handle text-based chat"""
    if not user_input or not user_input.strip():
        return "", chat_history, None
    
    chat_history, audio_path = process_user_input(user_input, chat_history, is_voice=False)
    return "", chat_history, audio_path

def voice_chat(audio_path, chat_history):
    """Handle voice-based chat with better error handling"""
    if audio_path is None:
        return chat_history, None

    user_input = audio_to_text(audio_path)
    if not user_input or not user_input.strip():
        error_msg = "I couldn't quite catch that. Could you try speaking a bit clearer?"
        chat_history.append(("🎤 [Audio unclear]", error_msg))
        error_audio = text_to_audio(error_msg)
        return chat_history, error_audio

    chat_history, response_audio = process_user_input(user_input, chat_history, is_voice=True)
    return chat_history, response_audio

def get_memory_summary():
    """Get a friendly memory summary"""
    if not conversation_memory:
        return "💭 Ready to start our conversation!"
    return f"💭 Remembering our last {len(conversation_memory)} exchanges"

def clear_memory():
    """Clear conversation memory"""
    global conversation_memory
    conversation_memory.clear()
    return "🧹 Memory cleared! Ready for a fresh start."

# Initialize components
print("🚀 Initializing CalmMe...")
llm = initialize_llm()

# Try to load or create vector database
qa_chain = None
try:
    if not os.path.exists("./chroma_db"):
        print("📚 Creating vector database...")
        vector_db = create_vector_db()
    else:
        print("📚 Loading existing vector database...")
        vector_db = Chroma(
            persist_directory='./chroma_db',
            embedding_function=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        )

    if vector_db and llm:
        def setup_qa_chain(vector_db, llm):
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})

            prompt_template = """You are a compassionate friend and emotional support companion. You have a warm, genuine personality and deep emotional intelligence.

Your conversation style:
- Talk naturally like a caring friend, not a therapist or chatbot
- Use everyday language with contractions (I'm, you're, can't, etc.)
- Show genuine empathy and emotional validation
- Share wisdom through relatable examples or gentle insights
- Remember and reference previous conversations naturally
- Sometimes use gentle humor when appropriate
- Ask thoughtful follow-up questions

Response guidelines:
✅ Provide emotional support and practical life advice
✅ Offer coping strategies and self-care suggestions
✅ Give relationship and personal growth guidance
✅ Validate emotions before offering solutions
✅ Use metaphors or stories to explain complex feelings
✅ Be encouraging but realistic

❌ Don't be clinical, robotic, or overly formal  
❌ Don't give medical diagnoses or prescriptions
❌ Don't lecture or sound preachy
❌ Don't start with artificial phrases like "I understand" or "Thank you for sharing"

Keep responses conversational (2-5 sentences typically) unless more detail is specifically needed.

Context from knowledge base: {context}
User's message: {question}

Respond as a caring friend who genuinely wants to help:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False
            )

        qa_chain = setup_qa_chain(vector_db, llm)
        print("✅ QA chain setup complete!")
    else:
        print("⚠️ Running in basic mode without vector database")

except Exception as e:
    print(f"⚠️ Vector database error: {e}. Running in basic mode.")

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="main-header">
        <h1 style="color: white; font-size: 3.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌟 CalmMe
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.3em; margin: 10px 0;">
            Your Compassionate AI Companion
        </p>
        <p style="color: rgba(255,255,255,0.8); font-size: 1em;">
            I'm here to listen, understand, and support you through everything ✨
        </p>
    </div>
    """)

    # Memory status indicator
    memory_status = gr.HTML("""<div class="memory-indicator">💭 Ready to start our conversation!</div>""")

    # Main chat interface
    with gr.Row():
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            container=True,
            bubble_full_width=False,
            elem_classes=["user-message", "bot-message"],
            avatar_images=("👤", "🤖")
        )

    # Audio output (hidden by default, shows when there's audio)
    audio_output = gr.Audio(
        label="🔊 Voice Response",
        autoplay=True,
        visible=False,
        show_download_button=False,
        interactive=False
    )

    # Input section
    with gr.Row():
        with gr.Column(scale=3):
            msg = gr.Textbox(
                label="💬 Share your thoughts...",
                placeholder="How are you feeling today? I'm here to listen and support you...",
                lines=3,
                elem_classes="custom-input",
                container=False,
                show_label=True
            )
        with gr.Column(scale=1):
            voice_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Voice Message",
                show_download_button=False,
                elem_classes="voice-input",
                container=False
            )

    # Control buttons
    with gr.Row():
        text_submit = gr.Button("💌 Send Message", elem_classes="custom-button", variant="primary", size="lg")
        voice_submit = gr.Button("🎙️ Send Voice", elem_classes="custom-button", variant="secondary", size="lg")
        clear = gr.Button("🔄 New Chat", elem_classes="custom-button", variant="secondary", size="lg")
        memory_clear = gr.Button("🧹 Clear Memory", elem_classes="custom-button", size="lg")

    # Features section
    gr.HTML("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-top: 25px; backdrop-filter: blur(10px);">
        <h3 style="color: white; text-align: center; margin-bottom: 20px;">✨ What Makes CalmMe Special</h3>
        <div class="feature-grid">
            <div class="feature-card">
                <div style="font-size: 2em; margin-bottom: 10px;">🧠</div>
                <div style="color: white; font-weight: bold;">Smart Memory</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.9em;">Remembers our conversations for personalized support</div>
            </div>
            <div class="feature-card">
                <div style="font-size: 2em; margin-bottom: 10px;">🎭</div>
                <div style="color: white; font-weight: bold;">Emotion Aware</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.9em;">Detects and responds to your emotional state</div>
            </div>
            <div class="feature-card">
                <div style="font-size: 2em; margin-bottom: 10px;">🗣️</div>
                <div style="color: white; font-weight: bold;">Voice & Text</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.9em;">Talk or type - whatever feels comfortable</div>
            </div>
            <div class="feature-card">
                <div style="font-size: 2em; margin-bottom: 10px;">🆘</div>
                <div style="color: white; font-weight: bold;">Crisis Support</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.9em;">Emergency resources when you need them most</div>
            </div>
        </div>
    </div>
    """)

    # Event handlers with better error handling
    def safe_update_memory():
        try:
            return get_memory_summary()
        except:
            return "💭 Memory system ready"

    # Text input handlers
    text_submit.click(
        fn=gradio_chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, audio_output],
        queue=True
    ).then(
        fn=lambda: safe_update_memory(),
        outputs=[memory_status],
        queue=False
    )

    msg.submit(
        fn=gradio_chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, audio_output],
        queue=True
    ).then(
        fn=lambda: safe_update_memory(),
        outputs=[memory_status],
        queue=False
    )

    # Voice input handler
    voice_submit.click(
        fn=voice_chat,
        inputs=[voice_input, chatbot],
        outputs=[chatbot, audio_output],
        queue=True
    ).then(
        fn=lambda: safe_update_memory(),
        outputs=[memory_status],
        queue=False
    )

    # Clear handlers
    clear.click(
        fn=lambda: ([], None, None),
        outputs=[chatbot, voice_input, audio_output],
        queue=False
    )

    memory_clear.click(
        fn=clear_memory,
        outputs=[memory_status],
        queue=False
    )

print("🌟 CalmMe is ready to help!")
print("💡 Features enabled:")
print(f"   🎤 Voice Input: {'✅' if stt_pipe else '❌'}")
print(f"   🔊 Voice Output: ✅")
print(f"   💭 Sentiment Analysis: {'✅' if sentiment_analyzer else '❌'}")
print(f"   🤖 AI Chat: {'✅' if qa_chain else '❌ (Basic mode)'}")
print(f"   📚 Knowledge Base: {'✅' if qa_chain else '❌'}")

if __name__ == "__main__":
    demo.launch(
        debug=True, 
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
