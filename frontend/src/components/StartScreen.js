import { useState, useEffect } from "react";
import { startConversation, getPersonas } from "../api";

export default function StartScreen({ onStart }) {
  const [personas, setPersonas] = useState({});
  const [persona, setPersona] = useState("");
  const [product, setProduct] = useState("cellular plans");
  const [scenario, setScenario] = useState("being overcharged");
  const [industry, setIndustry] = useState("telecom");
  const [loading, setLoading] = useState(true);

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

  const handleStart = async () => {
    try {
      const res = await startConversation(persona, {
        company_product: product,
        scenario_context: scenario,
        company_industry: industry
      });

      onStart(res);
    } catch (err) {
      console.error("Error starting conversation:", err);
      alert("Failed to start conversation. Make sure backend is running.");
    }
  };

  return (
    <div>
      <h1>Start Training</h1>

      <label>Choose Persona:</label>
      <br />

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

      <br /><br />

      <label>Product:</label>
      <br />
      <input
        type="text"
        value={product}
        onChange={(e) => setProduct(e.target.value)}
      />

      <br /><br />

      <label>Scenario:</label>
      <br />
      <input
        type="text"
        value={scenario}
        onChange={(e) => setScenario(e.target.value)}
      />

      <br /><br />

      <label>Industry:</label>
      <br />
      <input
        type="text"
        value={industry}
        onChange={(e) => setIndustry(e.target.value)}
      />

      <br /><br />

      <button onClick={handleStart} disabled={!persona}>
        Start Conversation
      </button>
    </div>
  );
}