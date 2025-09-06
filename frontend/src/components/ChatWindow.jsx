import React, { useState, useEffect, useRef } from "react";
import Message from "./Message";
import InputBox from "./InputBox";

function ChatWindow() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! I’m your Education Chatbot 📚. Ask me anything." },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom whenever messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSend = (userText) => {
    if (!userText.trim()) return;

    const newMessages = [...messages, { sender: "user", text: userText }];
    setMessages(newMessages);

    // Show typing indicator
    setIsTyping(true);

    setTimeout(() => {
      let botReply;
      if (userText.toLowerCase().includes("hello")) {
        botReply = "Hey there! 👋";
      } else if (userText.toLowerCase().includes("math")) {
        botReply = "Math is the study of numbers, shapes, and patterns 🔢.";
      } else if (userText.toLowerCase().includes("science")) {
        botReply = "Science helps us understand the natural world 🔬.";
      } else {
        botReply = "I'm still learning! Could you try rephrasing that? 🙂";
      }

      setMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
      setIsTyping(false);
    }, 1200); // simulate delay
  };

  return (
    <div className="chat-window">
      {/* Header */}
      <div className="chat-header">🎓 Education Chatbot</div>

      {/* Messages */}
      <div className="messages">
        {messages.map((msg, index) => (
          <Message key={index} sender={msg.sender} text={msg.text} />
        ))}

        {/* Typing indicator */}
        {isTyping && (
          <div className="message-row bot">
            <div className="message-bubble bot typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Box */}
      <InputBox onSend={handleSend} />
    </div>
  );
}

export default ChatWindow;
