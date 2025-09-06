# 🎓 Education Chatbot (React + Vite)

A simple **ChatGPT-style chatbot UI** built with **React + Vite**.  
It simulates an educational chatbot that responds to queries about **Math, Science**, and more.  

---

## 🚀 Features
- Dark theme inspired by ChatGPT
- Chat bubbles (left for bot, right for user)
- Sticky input bar at the bottom
- Typing indicator (animated dots) before bot replies
- Auto-scroll to latest message
- Simple rule-based responses (can be extended with an API)

---

## 🛠️ Installation & Setup

### 1. Install Node.js
Download and install **Node.js LTS** from [https://nodejs.org](https://nodejs.org).  
Make sure it’s available in your terminal:

```bash
node -v
npm -v

```
---
### Create Project with Vite
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm run dev
```
---
## Project Structure
frontend/
│── src/
│   ├── components/
│   │   ├── ChatWindow.jsx   # Main chat container
│   │   ├── Message.jsx      # Individual message bubble
│   │   └── InputBox.jsx     # Input bar for sending messages
│   ├── App.jsx              # Root component
│   ├── index.css            # Styles (ChatGPT-like)
│   └── main.jsx             # Entry point
│── package.json
│── README.md
