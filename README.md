# 🎓 AI Tutor Agent - Enhanced Learning Experience

An intelligent, personalized AI tutoring system that adapts to each student's learning level and provides comprehensive educational support through multiple modalities including text, voice, and interactive content.

## 🌟 Core Features

### 📚 **Multi-Modal Learning Support**
- **Text-based tutoring** with dynamic, personalized responses
- **Voice input/output** using OpenAI Whisper and TTS
- **PDF knowledge base** for content-specific learning
- **Interactive MCQ quizzes** with auto-grading
- **Progress tracking** with visual analytics

### 🎯 **Personalized AI Tutoring**
- **Dynamic instruction system** that adapts based on:
  - Student's current knowledge level (Beginner/Intermediate/Advanced)
  - Last test score (0-100 scale)
  - Total accumulated points
  - Earned badges and achievements
- **Context-aware responses** using uploaded study materials
- **Difficulty-appropriate content** generation

### 🔐 **Multi-Student System**
- **Individual student profiles** with SQLite database
- **Secure login/register** system
- **Personalized learning paths** for each student
- **Progress persistence** across sessions

### 📊 **Advanced Analytics & Gamification**
- **Real-time progress graphs** using matplotlib/seaborn
- **Achievement badges** system:
  - 🎯 Sharpshooter (90%+ accuracy)
  - 🎪 Great Performer (80%+ accuracy)
  - 📚 Good Student (70%+ accuracy)
  - 🏃 Marathon Runner (20+ attempts)
  - 🔥 Hot Streak (10+ correct answers)
- **Leaderboard system** with points tracking
- **PDF export** of learning reports

### 📄 **PDF Knowledge Base**
- **Upload and process** study materials (PDF format)
- **Content extraction** using PyPDF2 and LangChain
- **Vector store creation** with FAISS for semantic search
- **Context-aware question generation** from uploaded content
- **Hash-based caching** to avoid reprocessing

### 🧩 **MCQ Quiz System**
- **Auto-generation** of multiple-choice questions
- **Difficulty levels** (Easy/Medium/Hard)
- **Topic-based questions** from uploaded materials
- **Real-time scoring** and feedback
- **Performance tracking** per topic

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Clone the repository
git clone [your-repo-url]
cd AI-Tutor-Agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
OPENAI_API_KEY=your-openai-api-key
```

### **2. Run the Application**
```bash
# Start the application
streamlit run app.py
```

### **3. First Steps**
1. **Register** as a new student or **login** with existing credentials
2. **Upload study materials** in PDF format
3. **Start learning** through text or voice interactions
4. **Take quizzes** to test your knowledge
5. **Track progress** through the dashboard

## 🎯 **Usage Guide**

### **For Students:**
1. **Login/Register**: Create your personalized learning profile
2. **Upload PDFs**: Add your study materials to build a knowledge base
3. **Ask Questions**: Use text or voice to interact with your AI tutor
4. **Take Quizzes**: Test your knowledge with auto-generated questions
5. **Track Progress**: Monitor your learning journey with detailed analytics

### **For Educators:**
1. **Monitor Progress**: View detailed analytics for each student
2. **Customize Content**: Upload specific materials for different learning levels
3. **Generate Reports**: Export comprehensive learning reports as PDF

## 🔧 **Technical Architecture**

### **Core Components:**
- **Frontend**: Streamlit web interface
- **Backend**: Python with SQLite database
- **AI Models**: OpenAI GPT-4, Whisper, TTS
- **Data Processing**: LangChain, FAISS, PyPDF2
- **Visualization**: Matplotlib, Seaborn, Plotly

### **Database Schema:**
```sql
-- Students table
students(
    email TEXT PRIMARY KEY,
    name TEXT,
    level TEXT,
    last_score INTEGER,
    total_points INTEGER,
    badges TEXT
)

-- Progress tracking
progress(
    email TEXT,
    topic TEXT,
    score INTEGER,
    activity_type TEXT,
    timestamp DATETIME
)

-- PDF knowledge base
pdf_knowledge_base(
    email TEXT,
    filename TEXT,
    content_text TEXT,
    created_at DATETIME
)

-- MCQ questions
mcq_questions(
    email TEXT,
    question TEXT,
    options TEXT,
    correct_answer TEXT,
    topic TEXT
)
```

## 📈 **Performance Metrics**

### **Learning Effectiveness:**
- **Personalized difficulty** adaptation based on scores
- **Context-aware responses** using uploaded materials
- **Progress tracking** with visual feedback
- **Achievement system** for motivation

### **System Performance:**
- **Fast response times** with caching
- **Efficient PDF processing** with hash-based caching
- **Scalable architecture** for multiple students
- **Real-time updates** across all features

## 🛡️ **Security & Privacy**

- **Secure authentication** with password hashing
- **Individual data isolation** per student
- **No data sharing** between students
- **Local storage** with SQLite database
- **API key security** with environment variables

## 🔄 **Continuous Learning**

The system continuously adapts based on:
- **Student performance** in quizzes and interactions
- **Content engagement** patterns
- **Learning preferences** from voice/text usage
- **Progress trends** over time

## 📞 **Support & Feedback**

For issues, feature requests, or feedback:
- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Documentation**: Check the in-app help section
- **Community**: Join our Discord/Slack for discussions

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎓 AI Tutor Agent** - Making personalized learning accessible to everyone through intelligent AI tutoring.
