const BASE_URL = "http://127.0.0.1:5000";

export async function startConversation(persona_id, company_context) {
  const res = await fetch(`${BASE_URL}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ persona_id, company_context }),
  });

  if (!res.ok) {
  throw new Error("Request failed");
  }
  return res.json();
}

export async function sendMessage(convo_id, message) {
  const res = await fetch(`${BASE_URL}/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ convo_id, message }),
  });

  if (!res.ok) {
    throw new Error("Request failed");
  }
  return res.json();
}


export async function endConversation(convo_id) {
  const res = await fetch(`${BASE_URL}/end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ convo_id }),
  });

  if (!res.ok) {
    throw new Error("Request failed");
  }
  return res.json();
}

export async function getPersonas() {
  const res = await fetch(`${BASE_URL}/personas`);
  return res.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Upload failed");
  }

  return res.json();
}