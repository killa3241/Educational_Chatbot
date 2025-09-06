import React, { useState } from "react";

function InputBox({ onSend }) {
  const [input, setInput] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSend(input);
    setInput("");
  };

  return (
    <form className="input-box" onSubmit={handleSubmit}>
      <input
        type="text"
        value={input}
        placeholder="Ask me about math, science, or anything..."
        onChange={(e) => setInput(e.target.value)}
      />
      <button type="submit">➤</button>
    </form>
  );
}

export default InputBox;
