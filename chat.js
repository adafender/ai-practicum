import { useState } from "react";
import { startConversation, sendMessage, endConversation } from "../api";

export default function Chat() {
  const [convoId, setConvoId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [report, setReport] = useState(null);

  const start = async () => {
    const res = await startConversation(
      "gen_z_budget_student",
      {
        company_product: "a SaaS tool",
        scenario_context: "pricing inquiry",
        company_industry: "tech"
      }
    );

    setConvoId(res.convo_id);

    setMessages([
      { role: "assistant", content: res.first_message }
    ]);
  };

  const send = async () => {
    if (!input.trim()) return;

    const userMessage = input;

    setMessages(prev => [
      ...prev,
      { role: "user", content: userMessage }
    ]);

    setInput("");

    const res = await sendMessage(convoId, userMessage);

    setMessages(prev => [
      ...prev,
      { role: "assistant", content: res.response }
    ]);
  };

  const end = async () => {
    const res = await endConversation(convoId);
    setReport(res.report_card);
  };

  return (
    <div>
      <h1>AI Training Chat</h1>

      {!convoId && (
        <button onClick={start}>Start Conversation</button>
      )}

      <div style={{ border: "1px solid black", padding: "10px", height: "300px", overflowY: "scroll" }}>
        {messages.map((msg, i) => (
          <div key={i}>
            <b>{msg.role === "user" ? "You" : "Customer"}:</b> {msg.content}
          </div>
        ))}
      </div>

      {convoId && !report && (
        <>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button onClick={send}>Send</button>
          <button onClick={end}>End</button>
        </>
      )}

      {report && (
        <div>
          <h2>Report Card</h2>
          <pre>{report}</pre>
        </div>
      )}
    </div>
  );
}