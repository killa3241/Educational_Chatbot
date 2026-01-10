# Educational Chatbot - Web Interface Guide

## 🎓 Overview
This is a Flask-based web interface for the Educational Science Chatbot. It provides a user-friendly chat interface where students can ask science questions and receive:
1. **Text answers** - Generated using RAG (Retrieval-Augmented Generation)
2. **Relevant diagrams** - Automatically found and highlighted with query keywords

## 📋 Features

### ✨ Key Features:
- **Chat-like Interface** - Modern, responsive chatbot UI
- **Rules Display** - Instructions shown at startup
- **Parallel Processing** - Text and image processing happen simultaneously
- **Real-time Responses** - Loading indicators show processing status
- **Keyword Highlighting** - Relevant labels in diagrams are highlighted in green
- **Mark-based Answers** - Specify question marks for tailored responses

### 🎯 How It Works:
1. User enters a science question
2. System processes text answer using RAG from your textbook database
3. System finds most relevant diagram using semantic similarity
4. Both results are displayed: text first, then diagram with highlights

## 🚀 Setup Instructions

### Step 1: Activate Virtual Environment
```powershell
.\ragpart\Scripts\Activate.ps1
```

### Step 2: Install Flask
```powershell
pip install -r requirements_flask.txt
```

Or install directly:
```powershell
pip install Flask==3.0.0
```

### Step 3: Run the Flask Server
```powershell
python app.py
```

### Step 4: Open in Browser
Navigate to: **http://localhost:5000**

## 💬 How to Use the Chatbot

### Basic Usage:
1. **Ask Questions**: Type any science question in the input box
   - Example: "What is photosynthesis?"
   - Example: "Explain the structure of an atom"

2. **Specify Marks** (Optional): Add mark value for tailored answers
   - Example: "What is ecosystem? [2 marks]"
   - Example: "Explain respiration [5 marks]"

3. **View Results**:
   - Text answer appears first
   - Relevant diagram follows with highlighted keywords
   - Keywords from your question are shown in green boxes

### Rules & Guidelines:

📌 **Question Format**:
- Be clear and specific
- Use proper scientific terms
- Optional: Add "[X marks]" for mark-based answers

📌 **Mark Values**:
- [1 mark] - Very brief answer
- [2 marks] - Short answer
- [3 marks] - Medium answer
- [4 marks] - Detailed answer
- [5 marks] - Comprehensive answer

📌 **Response Format**:
- **Text Answer**: Generated from your textbook content
- **Diagram**: Most relevant image with keyword highlights
- **Keywords**: Words from your question matched in the diagram
- **Confidence**: How relevant the diagram is to your question

📌 **Visual Indicators**:
- 🟢 Green boxes = Matched keywords from your question
- ⬜ White boxes = Non-matching labels (blanked out)
- If no matches, all labels remain visible

## 📁 Project Structure

```
Educational_Chatbot/
├── app.py                      # Flask web application
├── combined_qa_system.py       # Core QA system (RAG + Image Mapping)
├── templates/
│   └── index.html             # Main webpage template
├── static/
│   ├── style.css              # Styling
│   └── script.js              # Frontend JavaScript
├── science_db/                # ChromaDB vector database
├── chapter5/                  # Source images folder
├── embeddings.pt              # Pre-computed image embeddings
└── requirements_flask.txt     # Flask dependencies
```

## 🔧 Technical Details

### Architecture:
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Backend**: Flask (Python)
- **RAG System**: ChromaDB + Ollama (llama3.2)
- **Image Mapping**: SentenceTransformers + EasyOCR
- **Processing**: Parallel execution using ThreadPoolExecutor

### API Endpoints:
- `GET /` - Main chatbot interface
- `POST /ask` - Process question and return results
- `GET /image/<filename>` - Serve processed images
- `GET /health` - System health check

### Response Format:
```json
{
  "question": "user question",
  "marks": "two",
  "timestamp": "02:30 PM",
  "text_answer": "generated answer",
  "text_success": true,
  "image_success": true,
  "image_path": "output_image.jpg",
  "image_name": "original_image.jpg",
  "confidence": "0.85",
  "query_words": ["ecosystem", "components"],
  "detected_words_count": 15,
  "matched_words_count": 3,
  "matches": [
    {"detected": "ecosystem", "query": "ecosystem", "score": "0.95"}
  ]
}
```

## 🎨 Interface Features

### Chat Interface:
- Clean, modern design with gradient header
- Responsive layout (works on mobile and desktop)
- Auto-scrolling to latest message
- Loading animations during processing
- Error handling with user-friendly messages

### Rules Section:
- Displayed on first load
- Can be reopened anytime using "Show Rules" button
- Contains detailed usage instructions
- Easy to dismiss with close button

### Input Section:
- Auto-resizing textarea
- Enter to send, Shift+Enter for new line
- Send button with hover effects
- Helpful hints below input

## ⚡ Performance Notes

- **Parallel Processing**: Text generation and image mapping run simultaneously
- **Loading Time**: First question may take 5-10 seconds (model initialization)
- **Subsequent Queries**: Typically 2-5 seconds
- **Image Processing**: OCR and keyword matching add ~1-2 seconds

## 🛠️ Troubleshooting

### Problem: Flask not installed
**Solution**: Run `pip install Flask==3.0.0`

### Problem: System not initialized
**Solution**: Ensure `science_db` folder and `embeddings.pt` exist

### Problem: Images not displaying
**Solution**: Check that `chapter5` folder contains source images

### Problem: Ollama not responding
**Solution**: Ensure Ollama is running: `ollama serve`

### Problem: Port 5000 already in use
**Solution**: Change port in `app.py`: `app.run(port=5001)`

## 📝 Example Questions

Try these questions to test the system:

**Basic Questions:**
- "What is photosynthesis?"
- "Explain the water cycle"
- "What are the parts of a flower?"

**With Marks:**
- "What is an ecosystem? [2 marks]"
- "Explain the digestive system [4 marks]"
- "Describe cellular respiration [5 marks]"

**Topic-Specific:**
- "What is the structure of an atom?"
- "How does the heart pump blood?"
- "What is the difference between acids and bases?"

## 🎯 Best Practices

1. **Clear Questions**: Use specific, clear language
2. **Scientific Terms**: Use proper terminology for better matching
3. **Mark Values**: Specify marks for exam-style answers
4. **Wait for Response**: Don't submit multiple questions rapidly
5. **Check Diagrams**: Look at highlighted keywords in images

## 🔒 Security Notes

- Input sanitization prevents XSS attacks
- Images served only from designated folders
- Error messages don't expose system details
- Flask debug mode should be disabled in production

## 📞 Support

If you encounter issues:
1. Check the console/terminal for error messages
2. Verify all dependencies are installed
3. Ensure database and embeddings exist
4. Check that Ollama is running

## 🎓 Educational Use

This system is designed for:
- **Students**: Study and exam preparation
- **Teachers**: Quick reference and teaching aid
- **Self-learners**: Understanding complex topics

---

**Made with ❤️ for educational purposes**

Press Ctrl+C in the terminal to stop the server when done.
