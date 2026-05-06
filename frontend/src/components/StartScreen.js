import { useState, useEffect } from "react";
import { startConversation, getPersonas, uploadDocument } from "../api";

export default function StartScreen({ onStart }) {
  const [personas, setPersonas] = useState({});
  const [persona, setPersona] = useState("");
  const [product, setProduct] = useState("cellular plans");
  const [scenario, setScenario] = useState("being overcharged");
  const [industry, setIndustry] = useState("telecom");
  const [loading, setLoading] = useState(true);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    async function loadPersonas() {
      try {
        const data = await getPersonas();
        setPersonas(data);

        const firstKey = Object.keys(data)[0];
        if (firstKey) setPersona(firstKey);
      } catch (err) {
        console.error("Failed to load personas:", err);
      } finally {
        setLoading(false);
      }
    }

    loadPersonas();
  }, []);

  const handleUpload = async () => {
  if (!file) {
    alert("Please select a file first");
    return;
  }

  try {
    setUploading(true);

    const res = await uploadDocument(file);
    alert(res.message || "Upload successful");

  } catch (err) {
    console.error("Upload failed:", err);
    alert("Upload failed");
  } finally {
    setUploading(false);
  }
};

  const handleStart = async () => {
    try {
      const res = await startConversation(persona, {
        company_product: product,
        scenario_context: scenario,
        company_industry: industry
      });

      // PLAY AUDIO HERE
      if (res.audio_url) {
        const audio = new Audio(`http://localhost:5000${res.audio_url}`);
        audio.play().catch(err => console.log("Audio blocked:", err));
      }

      onStart(res);

    } catch (err) {
      console.error("Error starting conversation:", err);
      alert("Failed to start conversation. Make sure backend is running.");
    }
  };

  return (
  <div className="card">
    <h2>Start Training</h2>

    {/* Persona */}
    <label>Choose Persona</label>
    {loading ? (
      <p>Loading personas...</p>
    ) : (
      <select value={persona} onChange={(e) => setPersona(e.target.value)}>
        {Object.entries(personas).map(([key, p]) => (
          <option key={key} value={key}>
            {p.name} ({p.personality?.difficulty})
          </option>
        ))}
      </select>
    )}

    {/* Product */}
    <label>Product</label>
    <input
      type="text"
      value={product}
      onChange={(e) => setProduct(e.target.value)}
    />

    {/* Scenario */}
    <label>Scenario</label>
    <input
      type="text"
      value={scenario}
      onChange={(e) => setScenario(e.target.value)}
    />

    {/* Industry */}
    <label>Industry</label>
    <input
      type="text"
      value={industry}
      onChange={(e) => setIndustry(e.target.value)}
    />

    {/* Upload */}
    <label>Upload Company Document (optional)</label>
    <input
      type="file"
      onChange={(e) => setFile(e.target.files[0])}
    />

    <button onClick={handleUpload} disabled={uploading}>
      {uploading ? "Uploading..." : "Upload Document"}
    </button>

    {/* Start Button */}
    <button onClick={handleStart} disabled={!persona}>
      Start Conversation
    </button>
  </div>
  );
}