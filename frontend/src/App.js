import { useState } from "react";
import StartScreen from "./components/StartScreen";
import ChatScreen from "./components/ChatScreen";
import ReportScreen from "./components/ReportScreen";
import "./App.css";
import "./components.css";

function App() {
  const [screen, setScreen] = useState("start");
  const [convoId, setConvoId] = useState(null);
  const [firstMessage, setFirstMessage] = useState("");
  const [report, setReport] = useState(null);

  const handleStart = (data) => {
    setConvoId(data.convo_id);
    setFirstMessage(data.first_message);
    setScreen("chat");
  };

  const handleEnd = (reportText) => {
    setReport(reportText);
    setScreen("report");
  };

  return (
    <div className="app-container">
      {/* HEADER */}
      <div className="header">
        <h1>SkillsTrAIner</h1>
        <span className="header-subtitle">
          {screen === "start" ? "Initialize Session" : screen === "chat" ? "Active Conversation" : "Performance Report"}
        </span>
      </div>

      {/* CONTENT WRAPPER */}
      <div className="content-wrapper">
        {/* LEFT PANEL */}
        <div className="left-panel">
          <StartScreen onStart={handleStart} />
        </div>

        {/* RIGHT PANEL */}
        <div className="right-panel">
          {screen === "start" && (
            <>
              <div className="info-card card">
                <h3>Welcome to SkillsTrAIner</h3>
                <p>
                  SkillsTrAIner is an AI-powered customer service training platform designed to help you improve your communication skills through realistic, interactive conversations. Practice handling difficult scenarios with intelligent AI personas, receive personalized feedback, and track your progress with detailed performance reports.
                </p>
              </div>
              <div className="placeholder">
                <h2>Start a Training Session</h2>
                <p>Configure your preferences on the left and begin.</p>
              </div>
            </>
          )}

          {screen === "chat" && (
            <ChatScreen
              convoId={convoId}
              firstMessage={firstMessage}
              onEnd={handleEnd}
            />
          )}

          {screen === "report" && (
            <ReportScreen report={report} />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
