import { useState } from "react";
import StartScreen from "./components/StartScreen";
import ChatScreen from "./components/ChatScreen";
import ReportScreen from "./components/ReportScreen";

function App() {
  const [screen, setScreen] = useState("start");
  const [convoId, setConvoId] = useState(null);
  const [firstMessage, setFirstMessage] = useState("");
  const [report, setReport] = useState(null);

  const handleStart = (data) => {
  console.log("START DATA:", data); // 🔥 DEBUG

  setConvoId(data.convo_id);
  setFirstMessage(data.first_message); // THIS is what you need
  setScreen("chat");
  };

  const handleEnd = (reportText) => {
    setReport(reportText);
    setScreen("report");
  };

  return (
    <div>
      {screen === "start" && <StartScreen onStart={handleStart} />}
      {screen === "chat" && (
        <ChatScreen 
          convoId={convoId} 
          firstMessage={firstMessage} 
          onEnd={handleEnd} 
        />
      )}
      {screen === "report" && <ReportScreen report={report} />}
    </div>
  );
}

export default App;