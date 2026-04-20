import { useState } from "react";
import { sendMessage, endConversation } from "../api";

export default function ChatScreen({ convoId, firstMessage, onEnd }) {
  const [messages, setMessages] = useState([
    { role: "customer", content: firstMessage }
  ]);

  const [input, setInput] = useState("");

  const handleSend = async () => {
    if (!input.trim()) return;

    if (input.trim().length < 5) {
      alert("Give a more complete response");
      return;
    }

    try {
      const res = await sendMessage(convoId, input);

      setMessages((prev) => [
        ...prev,
        { role: "agent", content: input },
        { role: "customer", content: res.response }
      ]);

      setInput("");
    } catch (err) {
      console.error(err);
      alert("Failed to send message");
    }
  };

  const handleEnd = async () => {
    try {
      const res = await endConversation(convoId);

      console.log("END RESPONSE:", res); // debug

      
      onEnd(res.report_card); 

    } catch (err) {
      console.error(err);
      alert("Failed to end conversation");
    }
  };

  return (
    <div>
      <h2>Chat</h2>

      <div style={{ height: "300px", overflowY: "scroll", border: "1px solid black" }}>
        {messages.map((msg, i) => (
          <div key={i}>
            <b>{msg.role === "agent" ? "You" : "Customer"}:</b> {msg.content}
          </div>
        ))}
      </div>

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your response..."
      />

      <button onClick={handleSend}>Send</button>
      <button onClick={handleEnd}>End Conversation</button>
    </div>
  );
}