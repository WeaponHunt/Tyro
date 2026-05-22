const els = {
  user: document.querySelector("#userInput"),
  language: document.querySelector("#languageSelect"),
  history: document.querySelector("#historyInput"),
  memoryToggle: document.querySelector("#memoryToggle"),
  messages: document.querySelector("#messages"),
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendBtn"),
  status: document.querySelector("#statusPill"),
  modelInfo: document.querySelector("#modelInfo"),
  clearHistory: document.querySelector("#clearHistoryBtn"),
  memoryText: document.querySelector("#memoryText"),
  addMemory: document.querySelector("#addMemoryBtn"),
  loadMemory: document.querySelector("#loadMemoryBtn"),
  memoryList: document.querySelector("#memoryList"),
};

const state = {
  busy: false,
  defaultUser: "default",
};

function settings() {
  return {
    user: (els.user.value || state.defaultUser || "default").trim(),
    language: els.language.value || "zh",
    history_rounds: Number.parseInt(els.history.value || "0", 10),
    use_memory: els.memoryToggle.checked,
  };
}

function persistSettings() {
  localStorage.setItem("talkrobot.web.settings", JSON.stringify(settings()));
}

function restoreSettings(config) {
  state.defaultUser = config.default_user || "default";
  const saved = JSON.parse(localStorage.getItem("talkrobot.web.settings") || "{}");
  els.user.value = saved.user || config.default_user || "default";
  els.language.value = saved.language || config.language || "zh";
  els.history.value = Number.isFinite(saved.history_rounds)
    ? saved.history_rounds
    : config.history_rounds || 5;
  els.memoryToggle.checked = saved.use_memory ?? true;
  els.modelInfo.textContent = `模型 ${config.model || "unknown"} · 本地 Web 服务已连接`;
}

function setStatus(text, mode = "") {
  els.status.textContent = text;
  els.status.className = `status-pill ${mode}`.trim();
}

function autosizeInput() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 180)}px`;
}

function avatarFor(role) {
  if (role === "assistant") {
    return '<img src="/static/tyro-mark.svg" alt="" />';
  }
  const name = settings().user || "U";
  return name.slice(0, 1).toUpperCase();
}

function appendMessage(role, text, meta) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = avatarFor(role);

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const metaEl = document.createElement("div");
  metaEl.className = "message-meta";
  metaEl.textContent = meta;

  const textEl = document.createElement("p");
  textEl.textContent = text;

  bubble.append(metaEl, textEl);
  article.append(avatar, bubble);
  els.messages.append(article);
  els.messages.scrollTop = els.messages.scrollHeight;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function sendMessage(message) {
  state.busy = true;
  els.send.disabled = true;
  setStatus("思考中", "busy");
  appendMessage("user", message, settings().user);

  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({ ...settings(), message }),
    });
    const details = [
      data.expression && data.expression !== "neutral" ? data.expression : null,
      data.used_memory ? "命中记忆" : null,
      data.used_history ? "使用上下文" : null,
      data.used_tools && data.used_tools.length ? `工具 ${data.used_tools.join(", ")}` : null,
      `${data.elapsed_ms} ms`,
    ].filter(Boolean);
    appendMessage("assistant", data.reply, details.length ? `Tyro · ${details.join(" · ")}` : "Tyro");
    if (data.memory_error) {
      setStatus("短期记忆模式", "busy");
    } else {
      setStatus("就绪");
    }
  } catch (error) {
    appendMessage("assistant", `出错了：${error.message}`, "Tyro");
    setStatus("请求失败", "error");
  } finally {
    state.busy = false;
    els.send.disabled = false;
    els.input.focus();
  }
}

async function addMemory() {
  const content = els.memoryText.value.trim();
  if (!content) {
    return;
  }
  setStatus("写入记忆", "busy");
  try {
    await requestJson("/api/memory", {
      method: "POST",
      body: JSON.stringify({ user: settings().user, content }),
    });
    els.memoryText.value = "";
    setStatus("记忆已添加");
    await loadMemories();
  } catch (error) {
    setStatus("记忆失败", "error");
    els.memoryList.innerHTML = `<div class="memory-item">添加失败：${error.message}</div>`;
  }
}

async function loadMemories() {
  setStatus("读取记忆", "busy");
  try {
    const user = encodeURIComponent(settings().user);
    const data = await requestJson(`/api/memories?user=${user}`);
    els.memoryList.innerHTML = "";
    if (!data.memories.length) {
      els.memoryList.innerHTML = '<div class="memory-item">暂无长期记忆</div>';
    } else {
      for (const item of data.memories.slice(0, 20)) {
        const div = document.createElement("div");
        div.className = "memory-item";
        div.textContent = item.text;
        els.memoryList.append(div);
      }
    }
    setStatus("就绪");
  } catch (error) {
    setStatus("读取失败", "error");
    els.memoryList.innerHTML = `<div class="memory-item">读取失败：${error.message}</div>`;
  }
}

async function clearHistory() {
  setStatus("清空上下文", "busy");
  try {
    await requestJson("/api/history/clear", {
      method: "POST",
      body: JSON.stringify(settings()),
    });
    setStatus("上下文已清空");
  } catch (error) {
    setStatus("清空失败", "error");
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) {
    return;
  }
  const message = els.input.value.trim();
  if (!message) {
    return;
  }
  els.input.value = "";
  autosizeInput();
  persistSettings();
  sendMessage(message);
});

els.input.addEventListener("input", autosizeInput);
els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

for (const control of [els.user, els.language, els.history, els.memoryToggle]) {
  control.addEventListener("change", persistSettings);
}

els.addMemory.addEventListener("click", addMemory);
els.loadMemory.addEventListener("click", loadMemories);
els.clearHistory.addEventListener("click", clearHistory);

async function init() {
  try {
    const config = await requestJson("/api/config");
    restoreSettings(config);
    setStatus("就绪");
  } catch (error) {
    els.modelInfo.textContent = "无法连接本地 Web 服务";
    setStatus("离线", "error");
  }
  autosizeInput();
}

init();
