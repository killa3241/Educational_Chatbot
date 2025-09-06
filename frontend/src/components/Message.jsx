import React from "react";

function Message({ sender, text }) {
  return (
    <div className={`message-row ${sender}`}>
      <div className={`message-bubble ${sender}`}>
        {text}
      </div>
    </div>
  );
}

export default Message;
