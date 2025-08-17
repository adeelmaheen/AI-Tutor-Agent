import os
import sqlite3
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from dataclasses import dataclass
from openai import OpenAI
import json
import time
from typing import List, Dict, Any
import warnings
from dotenv import load_dotenv
import hashlib
import pickle
import random
from datetime import datetime
# PDF and LangChain imports
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Suppress fpdf warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fpdf")
warnings.filterwarnings("ignore", message="You have both PyFPDF & fpdf2 installed")

# ---------- Setup ----------
st.set_page_config(page_title="AI Tutor Agent (Enhanced with PDF Features)", page_icon="🎓")

# Initialize OpenAI client with better error handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OpenAI API Key not found! Please set your OPENAI_API_KEY environment variable.")
    # st.info("💡 Set it in your terminal: `set OPENAI_API_KEY=your-key-here` (Windows) or `export OPENAI_API_KEY=your-key-here` (Mac/Linux)")
    

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize OpenAI client.")
    # st.stop()

# ---------- Enhanced Database ----------
def init_db():
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    
    # First, check if students table exists and get its structure
    c.execute("PRAGMA table_info(students)")
    columns = [row[1] for row in c.fetchall()]
    
    # Students table - handle migration for existing tables
    if not columns:  # Table doesn't exist
        c.execute("""
          CREATE TABLE students(
            email TEXT PRIMARY KEY,
            name TEXT,
            level TEXT,
            last_score INTEGER,
            assistant_id TEXT,
            thread_id TEXT,
            total_points INTEGER DEFAULT 0,
            badges TEXT DEFAULT '[]'
          )
        """)
    else:
        # Add missing columns if table exists but lacks new columns
        if 'total_points' not in columns:
            c.execute("ALTER TABLE students ADD COLUMN total_points INTEGER DEFAULT 0")
        if 'badges' not in columns:
            c.execute("ALTER TABLE students ADD COLUMN badges TEXT DEFAULT '[]'")
    
    # Progress tracking table - handle migration
    c.execute("PRAGMA table_info(progress)")
    progress_columns = [row[1] for row in c.fetchall()]
    
    if not progress_columns:  # Table doesn't exist
        c.execute("""
          CREATE TABLE progress(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            topic TEXT,
            score INTEGER,
            activity_type TEXT DEFAULT 'chat',
            FOREIGN KEY(email) REFERENCES students(email)
          )
        """)
    else:
        # Add activity_type column if it doesn't exist
        if 'activity_type' not in progress_columns:
            c.execute("ALTER TABLE progress ADD COLUMN activity_type TEXT DEFAULT 'chat'")
    
    # PDF Knowledge Base table - remove vector_store column to avoid pickle issues
    c.execute("""
      CREATE TABLE IF NOT EXISTS pdf_knowledge_base(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        filename TEXT,
        pdf_hash TEXT,
        content_text TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(email) REFERENCES students(email)
      )
    """)
    
    # MCQ Questions table
    c.execute("""
      CREATE TABLE IF NOT EXISTS mcq_questions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        pdf_id INTEGER,
        question TEXT,
        option_a TEXT,
        option_b TEXT,
        option_c TEXT,
        option_d TEXT,
        correct_answer TEXT,
        difficulty TEXT DEFAULT 'medium',
        topic TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(email) REFERENCES students(email),
        FOREIGN KEY(pdf_id) REFERENCES pdf_knowledge_base(id)
      )
    """)
    
    # MCQ Attempts table
    c.execute("""
      CREATE TABLE IF NOT EXISTS mcq_attempts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        question_id INTEGER,
        user_answer TEXT,
        is_correct BOOLEAN,
        points_earned INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(email) REFERENCES students(email),
        FOREIGN KEY(question_id) REFERENCES mcq_questions(id)
      )
    """)
    
    conn.commit()
    conn.close()

def get_student(email):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("SELECT email, name, level, last_score, assistant_id, thread_id, total_points, badges FROM students WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row

def upsert_student(email, name, level, last_score, assistant_id=None, thread_id=None):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO students 
                 (email, name, level, last_score, assistant_id, thread_id, total_points, badges) 
                 VALUES (?,?,?,?,?,?, 
                         COALESCE((SELECT total_points FROM students WHERE email=?), 0),
                         COALESCE((SELECT badges FROM students WHERE email=?), '[]'))""",
              (email, name, level, last_score, assistant_id, thread_id, email, email))
    conn.commit()
    conn.close()

def update_student_points_badges(email, points_to_add, new_badges=None):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    
    # Get current data
    c.execute("SELECT total_points, badges FROM students WHERE email=?", (email,))
    result = c.fetchone()
    if result:
        current_points, current_badges_json = result
        current_badges = json.loads(current_badges_json) if current_badges_json else []
        
        new_total_points = current_points + points_to_add
        if new_badges:
            current_badges.extend(new_badges)
            current_badges = list(set(current_badges))  # Remove duplicates
        
        c.execute("UPDATE students SET total_points=?, badges=? WHERE email=?",
                  (new_total_points, json.dumps(current_badges), email))
    
    conn.commit()
    conn.close()

def add_progress_entry(email, topic, score, activity_type='chat'):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("INSERT INTO progress (email, topic, score, activity_type) VALUES (?,?,?,?)",
              (email, topic, score, activity_type))
    conn.commit()
    conn.close()

def get_progress_history(email):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    
    # Check if activity_type column exists
    c.execute("PRAGMA table_info(progress)")
    columns = [row[1] for row in c.fetchall()]
    
    if 'activity_type' in columns:
        c.execute("SELECT timestamp, topic, score, activity_type FROM progress WHERE email=? ORDER BY timestamp", (email,))
    else:
        # Fallback for older database schema
        c.execute("SELECT timestamp, topic, score, 'chat' as activity_type FROM progress WHERE email=? ORDER BY timestamp", (email,))
    
    rows = c.fetchall()
    conn.close()
    return rows

# PDF Knowledge Base functions
def save_pdf_to_db(email, filename, pdf_text, pdf_hash):
    """Save PDF content to database without vector store (due to pickle issues)"""
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    
    # Just save the text content, we'll create vector store on demand
    c.execute("""INSERT INTO pdf_knowledge_base 
                 (email, filename, pdf_hash, content_text) 
                 VALUES (?,?,?,?)""",
              (email, filename, pdf_hash, pdf_text))
    
    conn.commit()
    pdf_id = c.lastrowid
    conn.close()
    return pdf_id

def get_pdf_by_hash(email, pdf_hash):
    """Get PDF content by hash"""
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("SELECT id, content_text FROM pdf_knowledge_base WHERE email=? AND pdf_hash=?",
              (email, pdf_hash))
    result = c.fetchone()
    conn.close()
    if result:
        pdf_id, content_text = result
        return pdf_id, content_text
    return None, None

def get_latest_pdf_content(email):
    """Get the latest PDF content for a student"""
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("SELECT content_text FROM pdf_knowledge_base WHERE email=? ORDER BY created_at DESC LIMIT 1",
              (email,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def save_mcq_to_db(email, pdf_id, question_data):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("""INSERT INTO mcq_questions 
                 (email, pdf_id, question, option_a, option_b, option_c, option_d, correct_answer, topic) 
                 VALUES (?,?,?,?,?,?,?,?,?)""",
              (email, pdf_id, question_data['question'], question_data['option_a'], 
               question_data['option_b'], question_data['option_c'], question_data['option_d'],
               question_data['correct_answer'], question_data['topic']))
    conn.commit()
    question_id = c.lastrowid
    conn.close()
    return question_id

def get_mcqs_for_student(email):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("""SELECT id, question, option_a, option_b, option_c, option_d, correct_answer, topic 
                 FROM mcq_questions WHERE email=? ORDER BY created_at DESC""", (email,))
    rows = c.fetchall()
    conn.close()
    return rows

def save_mcq_attempt(email, question_id, user_answer, is_correct, points_earned):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("""INSERT INTO mcq_attempts 
                 (email, question_id, user_answer, is_correct, points_earned) 
                 VALUES (?,?,?,?,?)""",
              (email, question_id, user_answer, is_correct, points_earned))
    conn.commit()
    conn.close()

def get_student_stats(email):
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    
    # Get total attempts and correct answers
    c.execute("""SELECT COUNT(*) as total_attempts, 
                        SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_answers,
                        SUM(points_earned) as total_points_earned
                 FROM mcq_attempts WHERE email=?""", (email,))
    stats = c.fetchone()
    
    # Get current points and badges
    c.execute("SELECT total_points, badges FROM students WHERE email=?", (email,))
    student_data = c.fetchone()
    
    conn.close()
    return stats, student_data

init_db()

# ---------- Models ----------
@dataclass
class StudentCtx:
    email: str
    name: str
    level: str
    last_score: int
    assistant_id: str = None
    thread_id: str = None
    total_points: int = 0
    badges: list = None

    def __post_init__(self):
        if self.badges is None:
            self.badges = []

# ---------- PDF Processing Functions ----------
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_vector_store(pdf_text):
    """Create FAISS vector store from PDF text"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def generate_mcqs_from_pdf(pdf_content, num_questions=5):
    """Generate MCQs from PDF content using direct text processing"""
    try:
        # Split content into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(pdf_content)
        
        # Use the first few chunks for question generation
        selected_chunks = chunks[:min(3, len(chunks))]  # Use first 3 chunks
        combined_text = "\n\n".join(selected_chunks)
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        mcqs = []
        for i in range(num_questions):
            try:
                prompt = f"""Based on the following text content, generate 1 multiple choice question:

TEXT CONTENT:
{combined_text[:2000]}  

Format your response EXACTLY as follows:

QUESTION: [Your question here]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
CORRECT: [A, B, C, or D]
TOPIC: [Main topic/subject of the question]

Make sure the question tests understanding of the provided content."""

                response = llm.invoke(prompt)
                
                # Parse the response
                mcq_data = parse_mcq_response(response.content)
                if mcq_data:
                    mcqs.append(mcq_data)
                    
            except Exception as e:
                st.error(f"Error generating MCQ {i+1}: {str(e)}")
                continue
        
        return mcqs
        
    except Exception as e:
        st.error(f"Error in MCQ generation: {str(e)}")
        return []

def parse_mcq_response(response):
    """Parse the MCQ response into structured data"""
    try:
        lines = response.strip().split('\n')
        mcq_data = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('QUESTION:'):
                mcq_data['question'] = line.replace('QUESTION:', '').strip()
            elif line.startswith('A)'):
                mcq_data['option_a'] = line.replace('A)', '').strip()
            elif line.startswith('B)'):
                mcq_data['option_b'] = line.replace('B)', '').strip()
            elif line.startswith('C)'):
                mcq_data['option_c'] = line.replace('C)', '').strip()
            elif line.startswith('D)'):
                mcq_data['option_d'] = line.replace('D)', '').strip()
            elif line.startswith('CORRECT:'):
                mcq_data['correct_answer'] = line.replace('CORRECT:', '').strip()
            elif line.startswith('TOPIC:'):
                mcq_data['topic'] = line.replace('TOPIC:', '').strip()
        
        # Validate that all required fields are present
        required_fields = ['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_answer']
        if all(field in mcq_data for field in required_fields):
            if 'topic' not in mcq_data:
                mcq_data['topic'] = 'General'
            return mcq_data
            
    except Exception as e:
        st.error(f"Error parsing MCQ: {str(e)}")
    
    return None

def calculate_pdf_hash(uploaded_file):
    """Calculate hash of uploaded file for caching"""
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(content).hexdigest()

# ---------- Badge System ----------
def check_and_award_badges(email, stats):
    """Check student performance and award badges"""
    new_badges = []
    total_attempts, correct_answers, _ = stats
    
    if total_attempts == 0:
        return new_badges
    
    accuracy = (correct_answers / total_attempts) * 100
    
    # Accuracy badges
    if accuracy >= 90 and total_attempts >= 5:
        new_badges.append("🎯 Sharpshooter")
    elif accuracy >= 80 and total_attempts >= 3:
        new_badges.append("🎪 Great Performer")
    elif accuracy >= 70 and total_attempts >= 3:
        new_badges.append("📚 Good Student")
    
    # Attempt badges
    if total_attempts >= 20:
        new_badges.append("🏃 Marathon Runner")
    elif total_attempts >= 10:
        new_badges.append("💪 Persistent Learner")
    elif total_attempts >= 5:
        new_badges.append("🌱 Getting Started")
    
    # Perfect streak badges
    if correct_answers >= 10:
        new_badges.append("🔥 Hot Streak")
    
    return new_badges

# ---------- OpenAI Agents SDK Functions ----------
def create_tutor_assistant(student: StudentCtx) -> str:
    """Create a personalized AI tutor assistant for the student"""
    return "chat_model"

def create_thread() -> str:
    """Create a new conversation thread"""
    return "local_thread"

def send_message_to_tutor(student: StudentCtx, message: str) -> str:
    """Send a message to the AI tutor and get response using chat completions"""
    try:
        # Check if student has uploaded PDF content for context
        pdf_content = get_latest_pdf_content(student.email)
        context_info = ""
        
        if pdf_content and len(pdf_content) > 100:
            # Use first 1000 characters of PDF as context
            context_info = f"\n\nStudent's Study Material Context:\n{pdf_content[:1000]}...\n"
        
        system_prompt = f"""
        You are a personalized AI tutor for {student.name}.
        
        Student Profile:
        - Name: {student.name}
        - Level: {student.level}
        - Last Score: {student.last_score}/100
        - Total Points: {student.total_points}
        - Badges: {', '.join(student.badges) if student.badges else 'None yet'}
        
        {context_info}
        
        Teaching Rules:
        - If score < 50: Use very simple language, break down concepts into small steps, provide basic examples
        - If score 50-80: Give step-by-step explanations, provide 2 practice questions after each concept
        - If score > 80: Provide advanced insights, challenging problems, and deeper conceptual understanding
        - If student has uploaded study material, prioritize answering based on that content when relevant
        - Reference the study material when applicable to make learning more relevant
        
        Always:
        - Be encouraging and supportive
        - Ask follow-up questions to check understanding
        - Provide real-world examples
        - Adapt your teaching style based on student responses
        - Acknowledge their achievements and badges when relevant
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def tts_speak(text: str) -> bytes:
    """Convert text to speech using OpenAI TTS"""
    try:
        audio = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text[:1000]
        )
        return audio.read()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return b""

def stt_transcribe(uploaded_audio) -> str:
    """Convert speech to text using OpenAI Whisper"""
    if uploaded_audio is None:
        return ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(uploaded_audio.getbuffer())
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        
        os.unlink(tmp_path)
        return transcript.text
    except Exception as e:
        st.error(f"STT Error: {e}")
        return ""

# ---------- UI: Authentication ----------
st.sidebar.title("🎓 Student Portal")

# Handle temporary mode changes from quick access buttons
if "temp_mode" in st.session_state and st.session_state.temp_mode:
    mode = st.session_state.temp_mode
    st.session_state.temp_mode = None
else:
    mode = st.sidebar.radio("Menu", ["Login", "Register", "PDF Knowledge Base", "MCQ Quiz", "Progress Dashboard"])

if mode == "Register":
    st.sidebar.subheader("📝 New Student Registration")
    r_email = st.sidebar.text_input("Email")
    r_name = st.sidebar.text_input("Name")
    r_level = st.sidebar.selectbox("Learning Level", ["Beginner", "Intermediate", "Advanced"])
    r_score = st.sidebar.slider("Current Knowledge Level (0-100)", 0, 100, 60)
    
    if st.sidebar.button("Register / Update Profile"):
        if r_email and r_name:
            with st.spinner("Creating your personalized AI tutor..."):
                student_ctx = StudentCtx(r_email, r_name, r_level, r_score)
                assistant_id = create_tutor_assistant(student_ctx)
                thread_id = create_thread()
                
                upsert_student(r_email, r_name, r_level, r_score, assistant_id, thread_id)
                st.sidebar.success("✅ Profile saved! Your AI tutor is ready. Go to Login.")
        else:
            st.sidebar.error("Please fill in all fields.")

# Session state initialization
if "student" not in st.session_state:
    st.session_state.student = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "current_mcqs" not in st.session_state:
    st.session_state.current_mcqs = []
if "current_mcq_index" not in st.session_state:
    st.session_state.current_mcq_index = 0

if mode == "Login":
    l_email = st.sidebar.text_input("Email to Login")
    if st.sidebar.button("🔐 Login"):
        row = get_student(l_email)
        if row:
            # Handle cases where the database might have missing columns
            email, name, level, last_score = row[0], row[1], row[2], row[3]
            assistant_id = row[4] if len(row) > 4 else None
            thread_id = row[5] if len(row) > 5 else None
            total_points = row[6] if len(row) > 6 else 0
            badges_json = row[7] if len(row) > 7 else '[]'
            
            try:
                badges_list = json.loads(badges_json) if badges_json else []
            except (json.JSONDecodeError, TypeError):
                badges_list = []
            
            st.session_state.student = StudentCtx(
                email=email, name=name, level=level, 
                last_score=last_score, assistant_id=assistant_id, thread_id=thread_id,
                total_points=total_points, badges=badges_list
            )
            st.sidebar.success(f"Welcome back, {name}! 👋")
            st.rerun()
        else:
            st.sidebar.error("❌ Student not found. Please register first.")

# ---------- PDF Knowledge Base ----------
if mode == "PDF Knowledge Base":
    st.title("📚 PDF Knowledge Base")
    
    if st.session_state.student is None:
        st.info("👈 Please login from the sidebar first!")
        st.stop()
    
    student = st.session_state.student
    st.success(f"📖 Building knowledge base for {student.name}")
    
    uploaded_file = st.file_uploader("Upload your study material (PDF)", type="pdf")
    
    if uploaded_file:
        pdf_hash = calculate_pdf_hash(uploaded_file)
        
        # Check if this PDF was already processed for this student
        existing_pdf_id, existing_vector_store = get_pdf_by_hash(student.email, pdf_hash)
        
        if existing_pdf_id:
            st.success("✅ This PDF was already processed! Knowledge base is ready.")
            st.info("🎯 Go to 'MCQ Quiz' to generate questions from this content.")
        else:
            with st.spinner("🔍 Processing PDF and building knowledge base..."):
                try:
                    # Extract text
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    
                    if len(pdf_text.strip()) < 100:
                        st.error("❌ PDF seems to have very little text. Please upload a proper document.")
                        st.stop()
                    
                    # Save to database (without vector store to avoid pickle issues)
                    pdf_id = save_pdf_to_db(student.email, uploaded_file.name, pdf_text, pdf_hash)
                    
                    st.success("✅ PDF processed successfully!")
                    st.info(f"📄 Filename: {uploaded_file.name}")
                    st.info(f"📝 Text Length: {len(pdf_text):,} characters")
                    
                    # Show preview of content
                    st.subheader("📖 Content Preview")
                    st.text_area("First 500 characters:", pdf_text[:500], height=150, disabled=True)
                    
                    # Award points for uploading PDF
                    update_student_points_badges(student.email, 50, ["📚 Knowledge Builder"])
                    add_progress_entry(student.email, "PDF Upload", 100, "pdf_upload")
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.error("Please try uploading a different PDF file.")

# ---------- MCQ Quiz ----------
if mode == "MCQ Quiz":
    st.title("🧩 MCQ Quiz Challenge")
    
    if st.session_state.student is None:
        st.info("👈 Please login from the sidebar first!")
        st.stop()
    
    student = st.session_state.student
    
    # Display current stats
    stats, student_data = get_student_stats(student.email)
    if stats and student_data:
        total_attempts, correct_answers, points_from_mcq = stats
        current_points, badges_json = student_data
        current_badges = json.loads(badges_json) if badges_json else []
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Total Points", current_points)
        with col2:
            st.metric("📝 Questions Attempted", total_attempts)
        with col3:
            st.metric("✅ Correct Answers", correct_answers)
        with col4:
            accuracy = (correct_answers / total_attempts * 100) if total_attempts > 0 else 0
            st.metric("🎪 Accuracy", f"{accuracy:.1f}%")
        
        if current_badges:
            st.write("🏅 **Your Badges:**", " | ".join(current_badges))
    
    # Check for available PDFs
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute("SELECT id, filename FROM pdf_knowledge_base WHERE email=?", (student.email,))
    available_pdfs = c.fetchall()
    conn.close()
    
    if not available_pdfs:
        st.warning("⚠️ No study materials found! Please upload a PDF in 'PDF Knowledge Base' first.")
        st.stop()
    
    # Generate new quiz
    if st.button("🎲 Generate New Quiz (5 Questions)"):
        # Get the latest PDF content
        pdf_content = get_latest_pdf_content(student.email)
        
        if pdf_content:
            with st.spinner("🤖 Generating questions from your study material..."):
                mcqs = generate_mcqs_from_pdf(pdf_content, num_questions=5)
                
                if mcqs:
                    # Save MCQs to database
                    for mcq in mcqs:
                        save_mcq_to_db(student.email, available_pdfs[-1][0], mcq)
                    
                    st.session_state.current_mcqs = mcqs
                    st.session_state.current_mcq_index = 0
                    st.success(f"✅ Generated {len(mcqs)} questions from your PDF! Start answering below.")
                    st.rerun()
                else:
                    st.error("❌ Failed to generate questions. Please try again or upload a different PDF.")
        else:
            st.error("❌ No PDF content found. Please upload a PDF first.")
    
    # Display current quiz
    if st.session_state.current_mcqs and st.session_state.current_mcq_index < len(st.session_state.current_mcqs):
        current_mcq = st.session_state.current_mcqs[st.session_state.current_mcq_index]
        question_num = st.session_state.current_mcq_index + 1
        
        st.subheader(f"Question {question_num}/{len(st.session_state.current_mcqs)}")
        st.write(f"**Topic:** {current_mcq.get('topic', 'General')}")
        
        # Display question and options
        st.write(f"**{current_mcq['question']}**")
        
        options = [
            f"A) {current_mcq['option_a']}",
            f"B) {current_mcq['option_b']}", 
            f"C) {current_mcq['option_c']}",
            f"D) {current_mcq['option_d']}"
        ]
        
        user_answer = st.radio("Select your answer:", options, key=f"mcq_{question_num}")
        
        if st.button("Submit Answer", key=f"submit_{question_num}"):
            selected_option = user_answer[0]  # Get A, B, C, or D
            correct_option = current_mcq['correct_answer'].strip()
            
            is_correct = selected_option == correct_option
            points_earned = 20 if is_correct else 0
            
            # Save attempt to database
            # Note: For this demo, we'll use a dummy question_id since we're not storing questions with proper IDs
            # save_mcq_attempt(student.email, 0, selected_option, is_correct, points_earned)
            
            if is_correct:
                st.success(f"✅ Correct! You earned {points_earned} points!")
                update_student_points_badges(student.email, points_earned)
                add_progress_entry(student.email, current_mcq.get('topic', 'Quiz'), 100, 'mcq_correct')
            else:
                st.error(f"❌ Wrong! The correct answer was {correct_option}) {current_mcq[f'option_{correct_option.lower()}']}")
                add_progress_entry(student.email, current_mcq.get('topic', 'Quiz'), 0, 'mcq_wrong')
            
            # Check for new badges
            stats, _ = get_student_stats(student.email)
            if stats:
                new_badges = check_and_award_badges(student.email, stats)
                if new_badges:
                    update_student_points_badges(student.email, 0, new_badges)
                    st.success(f"🎉 New badges earned: {', '.join(new_badges)}")
            
            # Move to next question
            st.session_state.current_mcq_index += 1
            
            if st.session_state.current_mcq_index >= len(st.session_state.current_mcqs):
                st.success("🎉 Quiz completed! Generate a new quiz to continue learning.")
                st.balloons()
            
            time.sleep(2)
            st.rerun()
    
    # Show recent MCQs from database
    st.subheader("📝 Your Recent Questions")
    recent_mcqs = get_mcqs_for_student(student.email)
    
    if recent_mcqs:
        for i, mcq in enumerate(recent_mcqs[:3]):  # Show last 3
            with st.expander(f"Question {i+1}: {mcq[7]} - {mcq[1][:50]}..."):
                st.write(f"**{mcq[1]}**")
                st.write(f"A) {mcq[2]}")
                st.write(f"B) {mcq[3]}")
                st.write(f"C) {mcq[4]}")
                st.write(f"D) {mcq[5]}")
                st.write(f"**Correct Answer:** {mcq[6]}")

# ---------- Progress Dashboard ----------
if mode == "Progress Dashboard" and st.session_state.student:
    st.title("📊 Progress Dashboard")
    student = st.session_state.student
    
    # Get comprehensive stats
    stats, student_data = get_student_stats(student.email)
    progress_data = get_progress_history(student.email)
    
    if student_data:
        current_points, badges_json = student_data
        current_badges = json.loads(badges_json) if badges_json else []
        
        # Header stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Total Points", current_points)
        with col2:
            if stats:
                total_attempts, correct_answers, _ = stats
                accuracy = (correct_answers / total_attempts * 100) if total_attempts > 0 else 0
                st.metric("🎪 Quiz Accuracy", f"{accuracy:.1f}%")
        with col3:
            st.metric("🏅 Badges Earned", len(current_badges))
        
        # Display badges
        if current_badges:
            st.subheader("🏆 Your Achievement Badges")
            badge_cols = st.columns(min(len(current_badges), 4))
            for i, badge in enumerate(current_badges):
                with badge_cols[i % 4]:
                    st.success(badge)
        
        # Progress over time
        if progress_data:
            df = pd.DataFrame(progress_data, columns=['Timestamp', 'Topic', 'Score', 'Activity'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            st.subheader("📈 Learning Progress Over Time")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Color code by activity type
                activity_colors = {
                    'chat': '#1f77b4',
                    'mcq_correct': '#2ca02c', 
                    'mcq_wrong': '#d62728',
                    'pdf_upload': '#ff7f0e'
                }
                
                for activity in df['Activity'].unique():
                    activity_data = df[df['Activity'] == activity]
                    ax.scatter(activity_data['Timestamp'], activity_data['Score'], 
                             label=activity.replace('_', ' ').title(), 
                             color=activity_colors.get(activity, '#1f77b4'),
                             s=100, alpha=0.7)
                
                ax.plot(df['Timestamp'], df['Score'], alpha=0.3, color='gray', linewidth=1)
                
                ax.set_xlabel('Date', fontweight='bold')
                ax.set_ylabel('Score (%)', fontweight='bold')
                ax.set_title(f'{student.name}\'s Learning Activities', fontweight='bold', pad=20)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Activity breakdown
                st.subheader("📊 Activity Summary")
                activity_counts = df['Activity'].value_counts()
                
                for activity, count in activity_counts.items():
                    activity_name = activity.replace('_', ' ').title()
                    if activity == 'mcq_correct':
                        st.success(f"✅ {activity_name}: {count}")
                    elif activity == 'mcq_wrong':
                        st.error(f"❌ MCQ Wrong: {count}")
                    elif activity == 'pdf_upload':
                        st.info(f"📚 {activity_name}: {count}")
                    else:
                        st.write(f"💬 {activity_name}: {count}")
        
        # Learning insights
        st.subheader("🧠 Learning Insights")
        if progress_data:
            # Topic performance
            topic_performance = {}
            for _, topic, score, activity in progress_data:
                if topic not in topic_performance:
                    topic_performance[topic] = []
                topic_performance[topic].append(score)
            
            insights = []
            strong_topics = []
            weak_topics = []
            
            for topic, scores in topic_performance.items():
                avg_score = sum(scores) / len(scores)
                if avg_score >= 80:
                    strong_topics.append(topic)
                elif avg_score < 60:
                    weak_topics.append(topic)
            
            col1, col2 = st.columns(2)
            with col1:
                if strong_topics:
                    st.success("🌟 **Strong Topics:**")
                    for topic in strong_topics[:3]:
                        st.write(f"• {topic}")
                
            with col2:
                if weak_topics:
                    st.warning("📖 **Topics to Focus On:**")
                    for topic in weak_topics[:3]:
                        st.write(f"• {topic}")
    
    else:
        st.info("🚀 Start learning to see your progress here!")

# ---------- Main Tutor Interface (Voice + Chat) ----------
if mode == "Login":
    st.title("🎓 AI Tutor Agent - Enhanced Learning Experience")
    
    if st.session_state.student is None:
        st.info("👈 Please login from the sidebar to start learning!")
        st.stop()
    
    student = st.session_state.student
    
    # Enhanced header with stats
    stats, student_data = get_student_stats(student.email)
    if student_data:
        current_points, badges_json = student_data
        current_badges = json.loads(badges_json) if badges_json else []
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 Student", student.name)
        with col2:
            st.metric("🎯 Points", current_points)
        with col3:
            st.metric("🏅 Badges", len(current_badges))
    
    st.success(f"🎯 Learning Level: {student.level} | Knowledge Score: {student.last_score}/100")
    
    # Quick access buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📚 Upload Study Material", type="secondary"):
            # Change to the PDF Knowledge Base mode using session state
            st.session_state.temp_mode = "PDF Knowledge Base"
            st.rerun()
    with col2:
        if st.button("🧩 Take Quiz", type="secondary"):
            st.session_state.temp_mode = "MCQ Quiz"
            st.rerun()
    with col3:
        if st.button("📊 View Progress", type="secondary"):
            st.session_state.temp_mode = "Progress Dashboard"
            st.rerun()
    
    st.markdown("---")
    
    # Display conversation history
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")
    
    # Voice Input Section
    st.subheader("🎤 Voice Learning")
    audio_input = st.audio_input("Record your question (click to start/stop)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗣️ Ask via Voice", type="primary"):
            if audio_input:
                with st.spinner("🎧 Transcribing your question..."):
                    question = stt_transcribe(audio_input)
                
                if question.strip():
                    st.success(f"🎯 You asked: {question}")
                    
                    # Add to conversation
                    st.session_state.conversation.append({"role": "user", "content": question})
                    
                    # Get AI tutor response
                    with st.spinner("🤖 Your AI tutor is thinking..."):
                        response = send_message_to_tutor(student, question)
                    
                    # Add progress entry
                    estimated_score = min(100, student.last_score + random.randint(-5, 15))
                    topic = question.split()[0] if question else "General"
                    add_progress_entry(student.email, topic, estimated_score)
                    
                    # Award points for interaction
                    update_student_points_badges(student.email, 5)  # 5 points per interaction
                    
                    # Generate audio response
                    with st.spinner("🔊 Generating voice response..."):
                        audio_response = tts_speak(response)
                    
                    # Add to conversation with audio
                    st.session_state.conversation.append({
                        "role": "assistant", 
                        "content": response,
                        "audio": audio_response
                    })
                    
                    st.rerun()
                else:
                    st.warning("⚠️ Could not transcribe audio. Please try again.")
            else:
                st.warning("⚠️ Please record audio first!")
    
    with col2:
        if st.button("🔄 Generate Study Questions"):
            # Get PDF content for context-aware question generation
            pdf_content = get_latest_pdf_content(student.email)
            
            if pdf_content:
                # Generate questions based on uploaded content
                context_prompt = f"""Based on the student's uploaded study material, generate 3 engaging study questions appropriate for a {student.level} level student with knowledge score {student.last_score}/100. 

Study Material Context:
{pdf_content[:1500]}

Make the questions:
1. Relevant to the uploaded content
2. Appropriate for their learning level
3. Engaging and educational
4. Include brief explanations or hints"""
            else:
                # Fallback to general questions
                context_prompt = f"Generate 3 study questions appropriate for a {student.level} level student with knowledge score {student.last_score}/100. Make them engaging and educational."
            
            with st.spinner("🎓 Generating personalized study questions..."):
                study_questions = send_message_to_tutor(student, context_prompt)
            
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"📝 **Study Questions Based On Your Material:**\n\n{study_questions}"
            })
            
            add_progress_entry(student.email, "Study Questions", 90, "study_generation")
            update_student_points_badges(student.email, 10)
            
            st.rerun()
    
    # Text Input
    text_question = st.chat_input("💬 Type your question here...")
    if text_question:
        # Add user message
        st.session_state.conversation.append({"role": "user", "content": text_question})
        
        # Get AI response
        with st.spinner("🤖 AI Tutor is responding..."):
            response = send_message_to_tutor(student, text_question)
        
        # Add progress entry
        estimated_score = min(100, student.last_score + random.randint(-5, 15))
        topic = text_question.split()[0] if text_question else "General"
        add_progress_entry(student.email, topic, estimated_score)
        
        # Award points
        update_student_points_badges(student.email, 5)
        
        # Generate audio
        audio_response = tts_speak(response)
        
        # Add assistant message
        st.session_state.conversation.append({
            "role": "assistant", 
            "content": response,
            "audio": audio_response
        })
        
        st.rerun()
    
    # Export and utility options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Learning Tools")
    
    if st.sidebar.button("📄 Generate Learning Report"):
        if st.session_state.conversation:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                
                # Title
                pdf.set_font("Helvetica", 'B', 16)
                pdf.cell(0, 10, f"Learning Session - {student.name}", 
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                pdf.ln(10)
                
                # Add student stats
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 8, f"Student: {student.name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.cell(0, 8, f"Level: {student.level}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                if student_data:
                    current_points, badges_json = student_data
                    pdf.cell(0, 8, f"Total Points: {current_points}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(10)
                
                # Conversation
                pdf.set_font("Helvetica", size=11)
                for msg in st.session_state.conversation:
                    role = "Student" if msg["role"] == "user" else "AI Tutor"
                    pdf.set_font("Helvetica", 'B', 11)
                    pdf.cell(0, 8, role, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font("Helvetica", size=10)
                    
                    # Clean content for PDF
                    content = msg["content"]
                    content = ''.join(char for char in content if ord(char) < 256)
                    pdf.multi_cell(0, 6, content)
                    pdf.ln(3)
                
                # Generate PDF
                pdf_content = bytes(pdf.output())
                
                st.sidebar.download_button(
                    "⬇️ Download Learning Report", 
                    pdf_content, 
                    file_name=f"learning_session_{student.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
                st.sidebar.success("✅ PDF generated successfully!")
                
            except Exception as e:
                st.sidebar.error(f"❌ PDF generation failed: {str(e)}")
        else:
            st.sidebar.info("Start a conversation to generate a report!")
    
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.conversation = []
        st.rerun()
    
    # Study streak tracker
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔥 Study Stats")
    
    # Get progress data for sidebar stats
    progress_data = get_progress_history(student.email)
    if progress_data:
        today = datetime.now().date()
        recent_dates = set()
        for timestamp, _, _, _ in progress_data[-7:]:  # Last 7 entries
            try:
                entry_date = datetime.fromisoformat(timestamp).date()
                recent_dates.add(entry_date)
            except (ValueError, TypeError):
                # Handle different timestamp formats
                continue
        
        streak_days = len([d for d in recent_dates if (today - d).days <= 7])
        st.sidebar.metric("📅 Active Days (Last Week)", streak_days)
        
        if streak_days >= 5:
            st.sidebar.success("🔥 You're on fire! Great consistency!")
        elif streak_days >= 3:
            st.sidebar.info("👍 Good study rhythm!")
        else:
            st.sidebar.warning("📖 Try to study more regularly!")
    else:
        st.sidebar.info("📚 Start learning to track your progress!")

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.markdown("🎓 **Enhanced AI Tutor Agent**")
st.sidebar.markdown("✨ Features: PDF Knowledge Base, MCQ Generation, Voice Learning, Progress Tracking")
st.sidebar.markdown("🔗 Powered by OpenAI GPT-4, Whisper & TTS")