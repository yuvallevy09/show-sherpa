const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

async function apiRequest(path, { method = "GET", body } = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    credentials: "include",
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${method} ${path} failed (${res.status}): ${text || res.statusText}`);
  }

  // Some endpoints may return no content
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return null;
  return await res.json();
}

const LS_USER_KEY = "showsherpa_user_v1";
const LS_CONV_KEY = "showsherpa_conversations_v1";

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function nowIso() {
  return new Date().toISOString();
}

function newId() {
  // crypto.randomUUID is supported in modern browsers; fallback for older ones.
  return (globalThis.crypto && "randomUUID" in globalThis.crypto)
    ? globalThis.crypto.randomUUID()
    : `id_${Math.random().toString(16).slice(2)}_${Date.now()}`;
}

// NOTE: Conversations are still local-first for now. Next incremental step is to
// move this store to FastAPI endpoints.
const ConversationStore = {
  list(order = "-last_message_at") {
    const conversations = loadJson(LS_CONV_KEY, []);
    const sorted = [...conversations].sort((a, b) => {
      const av = a?.last_message_at || "";
      const bv = b?.last_message_at || "";
      return order.startsWith("-") ? bv.localeCompare(av) : av.localeCompare(bv);
    });
    return Promise.resolve(sorted);
  },

  create(data) {
    const conversations = loadJson(LS_CONV_KEY, []);
    const created = {
      id: newId(),
      title: data?.title || "New conversation",
      messages: data?.messages || [],
      last_message_at: data?.last_message_at || nowIso(),
      ...data,
    };
    conversations.push(created);
    saveJson(LS_CONV_KEY, conversations);
    return Promise.resolve(created);
  },

  update(id, data) {
    const conversations = loadJson(LS_CONV_KEY, []);
    const idx = conversations.findIndex((c) => c.id === id);
    if (idx === -1) return Promise.reject(new Error(`Conversation not found: ${id}`));
    conversations[idx] = { ...conversations[idx], ...data };
    saveJson(LS_CONV_KEY, conversations);
    return Promise.resolve(conversations[idx]);
  },

  delete(id) {
    const conversations = loadJson(LS_CONV_KEY, []);
    const next = conversations.filter((c) => c.id !== id);
    saveJson(LS_CONV_KEY, next);
    return Promise.resolve({ ok: true });
  },
};

export const api = {
  auth: {
    async me() {
      // Prefer backend, but fall back to local cached user if backend isn't running yet.
      try {
        const user = await apiRequest("/me");
        saveJson(LS_USER_KEY, user);
        return user;
      } catch (e) {
        const cached = loadJson(LS_USER_KEY, null);
        if (cached) return cached;
        throw e;
      }
    },

    async updateMe(patch) {
      const user = await apiRequest("/me", { method: "PATCH", body: patch || {} });
      saveJson(LS_USER_KEY, user);
      return user;
    },

    async logout() {
      localStorage.removeItem(LS_USER_KEY);
      // keep conversations by default; comment next line if you want to preserve them
      // localStorage.removeItem(LS_CONV_KEY);
      window.location.reload();
    },
  },

  entities: {
    Conversation: ConversationStore,
  },

  // Integrations will be reintroduced as backend endpoints if needed.
  integrations: {},
};


