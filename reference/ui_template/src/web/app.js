// Movie Lab 웹 화면 (F2013). 서버(src/server.py)의 JSON API만 호출한다.
// API 키는 서버 환경변수에만 있고 이 파일에는 없다.

const state = {
  movies: [],                 // [{movie_id, label}]
  idByLabel: new Map(),
  labelById: new Map(),
  movieId: "",
  histories: loadHistories(), // {movie_id: [{role, content, sources}]}
  busy: false,
};

const $ = (id) => document.getElementById(id);

// ── 저장 (새로고침해도 대화가 남도록. 저장이 막힌 브라우저에서도 동작은 한다) ──
function loadHistories() {
  try {
    return JSON.parse(localStorage.getItem("movielab.histories") || "{}");
  } catch {
    return {};
  }
}

function saveHistories() {
  try {
    localStorage.setItem("movielab.histories", JSON.stringify(state.histories));
  } catch {
    /* 저장이 막혀 있으면 이번 세션 동안만 유지한다 */
  }
}

function history(movieId) {
  if (!state.histories[movieId]) state.histories[movieId] = [];
  return state.histories[movieId];
}

// ── API ────────────────────────────────────────────────────────────────────
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let data = null;
  try {
    data = await response.json();
  } catch {
    /* 본문이 JSON이 아닌 오류 */
  }
  if (!response.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : "요청을 처리하지 못했습니다.";
    throw new Error(detail);
  }
  return data;
}

const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });

// ── 안전한 마크다운 표시 ─────────────────────────────────────────────────────
// 먼저 HTML을 이스케이프하고 나서 허용한 표기(제목·굵게·목록·근거 번호)만 태그로 바꾼다.
// 답변을 innerHTML로 바로 넣으면 문서에 섞인 HTML이 그대로 실행될 수 있다.
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function inline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/[\[［]\s*근거[^\]］<]*[\]］]/g, (match) => `<span class="cite">${match}</span>`);
}

function renderMarkdown(markdown) {
  const lines = escapeHtml(markdown).split(/\r?\n/);
  const html = [];
  let list = false;
  let paragraph = [];
  const flush = () => {
    if (paragraph.length) html.push(`<p>${inline(paragraph.join("<br>"))}</p>`);
    paragraph = [];
  };
  for (const raw of lines) {
    const line = raw.trim();
    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (!bullet && list) {
      html.push("</ul>");
      list = false;
    }
    if (!line) {
      flush();
    } else if (/^#{1,4}\s+/.test(line)) {
      flush();
      html.push(`<h4>${inline(line.replace(/^#{1,4}\s+/, ""))}</h4>`);
    } else if (bullet) {
      flush();
      if (!list) {
        html.push("<ul>");
        list = true;
      }
      html.push(`<li>${inline(bullet[1])}</li>`);
    } else {
      paragraph.push(line);
    }
  }
  flush();
  if (list) html.push("</ul>");
  return html.join("");
}

// ── 근거 목록 (F2008 · F2016) ────────────────────────────────────────────────
function mergeSources(sources) {
  // 한 문서가 여러 청크로 나뉘어 같은 제목이 반복되므로 한 줄로 합치고 번호는 모두 남긴다.
  const merged = new Map();
  for (const source of sources) {
    const key = `${source.title}|${source.source_type}|${source.url}`;
    if (!merged.has(key)) merged.set(key, { ...source, indexes: [] });
    merged.get(key).indexes.push(source.index);
  }
  return [...merged.values()];
}

function sourcesElement(message) {
  // 서버는 답변이 실제로 인용한 근거만 sources에 담고, 인용 여부를 cited로 알려준다. (F2008)
  // cited가 없는 예전 대화 기록은 본문의 근거 표기로 대신 판단한다.
  const sources = message.sources || [];
  const cited = typeof message.cited === "boolean" ? message.cited : /[\[［]\s*근거\s*\d/.test(message.content);
  if (!cited || !sources.length) {
    const note = document.createElement("p");
    note.className = "caption";
    note.textContent = "ⓘ 답변에 인용된 근거가 없습니다.";
    return note;
  }
  const documents = mergeSources(sources);
  const box = document.createElement("details");
  box.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `답변에 사용한 근거 ${documents.length}건`;
  box.appendChild(summary);

  const list = document.createElement("ul");
  for (const doc of documents) {
    const item = document.createElement("li");
    const numbers = doc.indexes.filter((n) => n != null).sort((a, b) => a - b).join(", ");
    const label = document.createElement("b");
    label.textContent = numbers ? `[근거 ${numbers}] ` : "";
    item.appendChild(label);
    item.appendChild(document.createTextNode(doc.title));
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = doc.source_label;
    item.appendChild(badge);
    if (doc.url && /^https?:\/\//.test(doc.url)) {
      const link = document.createElement("a");
      link.href = doc.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "원문";
      item.appendChild(link);
    }
    list.appendChild(item);
  }
  box.appendChild(list);
  return box;
}

// ── 대화 화면 ────────────────────────────────────────────────────────────────
function messageElement(message) {
  const box = document.createElement("div");
  box.className = `message ${message.role === "user" ? "user" : "assistant"}`;
  const role = document.createElement("div");
  role.className = "role";
  role.textContent = message.role === "user" ? "나" : "Movie Lab";
  box.appendChild(role);

  const body = document.createElement("div");
  if (message.role === "user") {
    body.textContent = message.content; // 사용자 입력은 표시만 한다
  } else {
    body.innerHTML = renderMarkdown(message.content);
  }
  box.appendChild(body);

  const sources = message.role === "assistant" ? sourcesElement(message) : null;
  if (sources) box.appendChild(sources);
  return box;
}

function renderChat() {
  const hasMovie = Boolean(state.movieId);
  $("empty-guide").classList.toggle("hidden", hasMovie);
  $("movie-title").classList.toggle("hidden", !hasMovie);
  $("chat-form").classList.toggle("hidden", !hasMovie);
  $("movie-info").classList.toggle("hidden", !hasMovie);

  const container = $("messages");
  container.replaceChildren();
  if (!hasMovie) return;
  $("movie-title").textContent = state.labelById.get(state.movieId) || state.movieId;
  for (const message of history(state.movieId)) container.appendChild(messageElement(message));
  window.scrollTo({ top: document.body.scrollHeight });
}

function renderTalking() {
  const talking = Object.entries(state.histories).filter(([, items]) => items.length);
  $("talking-box").classList.toggle("hidden", !talking.length);
  const list = $("talking-list");
  list.replaceChildren();
  for (const [movieId, items] of talking) {
    const item = document.createElement("li");
    const asked = items.filter((m) => m.role === "user").length;
    const current = movieId === state.movieId;
    item.className = current ? "current" : "";
    item.textContent = `${current ? "▶" : "·"} ${state.labelById.get(movieId) || movieId} (${asked}문답)`;
    item.addEventListener("click", () => selectMovie(movieId));
    list.appendChild(item);
  }
  $("clear-button").classList.toggle("hidden", !(state.movieId && history(state.movieId).length));
}

function showError(text) {
  const box = document.createElement("div");
  box.className = "notice error";
  box.textContent = text;
  $("messages").appendChild(box);
}

function setBusy(busy, text = "") {
  state.busy = busy;
  $("progress").classList.toggle("hidden", !busy);
  $("progress-text").textContent = text;
  for (const id of ["chat-input", "send-button", "review-button", "movie-input"]) $(id).disabled = busy;
}

// ── 영화 선택 (F2009) ────────────────────────────────────────────────────────
async function selectMovie(movieId) {
  state.movieId = movieId;
  $("movie-input").value = state.labelById.get(movieId) || "";
  $("recommend-view").classList.add("hidden");
  $("chat-view").classList.remove("hidden");
  $("recommend-panel").open = false;
  renderChat();
  renderTalking();
  await loadMovieInfo(movieId);
}

async function loadMovieInfo(movieId) {
  try {
    const movie = await api(`/api/movies/${encodeURIComponent(movieId)}`);
    if (movieId !== state.movieId) return;
    $("movie-director").textContent = `감독: ${movie.director.join(", ")}`;
    $("movie-cast").textContent = `출연: ${movie.cast.slice(0, 3).join(", ")}`;
    const docs = [...movie.wiki_titles];
    $("movie-docs").textContent = docs.length
      ? `수집된 위키 문서 ${docs.length}건${movie.wikidata ? " · 위키데이터 " + movie.wikidata : ""}\n${docs.join(" · ")}`
      : "아직 수집된 문서가 없습니다. 첫 질문 때 수집합니다.";

    const similar = $("similar-list");
    similar.replaceChildren();
    for (const other of movie.similar) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = other.label;
      button.addEventListener("click", () => selectMovie(other.movie_id));
      similar.appendChild(button);
    }
    $("similar-box").classList.toggle("hidden", !movie.similar.length);
  } catch (error) {
    $("movie-docs").textContent = error.message;
  }
}

// ── 질의응답 (F2007 · F2012) ─────────────────────────────────────────────────
async function ask(question) {
  const movieId = state.movieId;
  const messages = history(movieId);
  messages.push({ role: "user", content: question });
  renderChat();
  renderTalking();
  try {
    setBusy(true, "관련 위키백과·위키데이터 문서를 확인하고 있습니다...");
    const collected = await post(`/api/movies/${encodeURIComponent(movieId)}/collect`, { question });
    const added = collected.added_titles.length;
    setBusy(true, `${added ? `위키 문서 ${added}건 추가` : `수집된 위키 문서 ${collected.titles.length}건 사용`} · 문서를 검색하고 답변을 만드는 중...`);
    const result = await post("/api/ask", { movie_id: movieId, question });
    messages.push({ role: "assistant", content: result.answer, sources: result.sources, cited: result.cited });
    saveHistories();
    if (movieId === state.movieId) {
      renderChat();
      loadMovieInfo(movieId);
    }
  } catch (error) {
    messages.pop(); // 답이 없는 질문은 기록에 남기지 않는다
    renderChat();
    showError(`답변을 만들지 못했습니다. ${error.message}`);
  } finally {
    setBusy(false);
    renderTalking();
  }
}

// ── 관람 가이드 (F2010) ──────────────────────────────────────────────────────
async function writeReview() {
  const movieId = state.movieId;
  if (!movieId || state.busy) return;
  try {
    setBusy(true, "관련 문서를 확인하고 리뷰를 작성하는 중...");
    await post(`/api/movies/${encodeURIComponent(movieId)}/collect`, { question: "" });
    const result = await post(`/api/movies/${encodeURIComponent(movieId)}/review`);
    history(movieId).push({ role: "assistant", content: result.review, sources: result.sources, cited: result.cited });
    saveHistories();
    if (movieId === state.movieId) renderChat();
  } catch (error) {
    showError(`리뷰를 만들지 못했습니다. ${error.message}`);
  } finally {
    setBusy(false);
    renderTalking();
  }
}

// ── 조건 기반 탐색 (F2011) ───────────────────────────────────────────────────
async function findMovies(query) {
  $("chat-view").classList.add("hidden");
  $("recommend-view").classList.remove("hidden");
  $("recommend-title").textContent = `"${query}" 추천 결과`;
  const results = $("recommend-results");
  results.replaceChildren();
  const waiting = document.createElement("div");
  waiting.className = "progress";
  waiting.innerHTML = '<span class="spinner"></span>';
  waiting.appendChild(document.createTextNode("영화를 찾는 중..."));
  results.appendChild(waiting);

  try {
    const data = await post("/api/recommend", { query });
    results.replaceChildren();
    if (!data.results.length) {
      const empty = document.createElement("div");
      empty.className = "notice info";
      empty.textContent = "요청에 맞는 영화를 찾지 못했습니다. 다른 표현으로 다시 찾아보세요.";
      results.appendChild(empty);
    }
    for (const movie of data.results) {
      const card = document.createElement("div");
      card.className = "card";
      const title = document.createElement("h4");
      title.textContent = `🎬 ${movie.label}`;
      const meta = document.createElement("p");
      meta.className = "caption";
      meta.textContent = `${movie.genres.join(", ")} · 감독 ${movie.director.join(", ")}`;
      const reason = document.createElement("p");
      reason.textContent = movie.reason;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "이 영화에 대해 질문하기";
      button.addEventListener("click", () => selectMovie(movie.movie_id));
      card.append(title, meta, reason, button);
      results.appendChild(card);
    }
  } catch (error) {
    results.replaceChildren();
    const box = document.createElement("div");
    box.className = "notice error";
    box.textContent = `영화를 찾지 못했습니다. ${error.message}`;
    results.appendChild(box);
  }
}

// ── 시작 ───────────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const status = await api("/api/status");
    const idx = status.indexes;
    $("index-status").textContent =
      `색인: 영화 ${idx.movie_meta}건 · 위키백과 ${idx.wikipedia}청크 · 위키데이터 ${idx.wikidata}청크 (모델 ${status.chat_model})`;
    const warnings = [];
    if (!status.api_key) warnings.push("GMS_KEY가 설정되지 않았습니다. src/.env를 확인하세요.");
    if (!idx.movie_meta) warnings.push("영화 색인이 없습니다. python src/index_movies.py 를 실행하세요.");
    if (!idx.wikipedia) warnings.push("위키 색인이 비어 있습니다. 사실 질문이 모두 '문서로 확인 불가'로 나옵니다. python src/repair_wiki_index.py --apply 를 실행하세요.");
    $("setup-warning").textContent = warnings.join(" ");
    $("setup-warning").classList.toggle("hidden", !warnings.length);
  } catch (error) {
    $("index-status").textContent = `서버 상태를 확인하지 못했습니다. ${error.message}`;
  }
}

async function loadMovies() {
  state.movies = await api("/api/movies");
  const list = $("movie-list");
  for (const movie of state.movies) {
    state.idByLabel.set(movie.label, movie.movie_id);
    state.labelById.set(movie.movie_id, movie.label);
    const option = document.createElement("option");
    option.value = movie.label;
    list.appendChild(option);
  }
}

function bindEvents() {
  $("movie-input").addEventListener("change", (event) => {
    const movieId = state.idByLabel.get(event.target.value.trim());
    if (movieId) selectMovie(movieId);
  });

  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("chat-input").value.trim();
    if (!question || state.busy || !state.movieId) return; // 빈 질문은 보내지 않는다
    $("chat-input").value = "";
    ask(question);
  });

  $("recommend-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const query = $("recommend-input").value.trim();
    if (query) findMovies(query);
  });

  $("back-button").addEventListener("click", () => {
    $("recommend-view").classList.add("hidden");
    $("chat-view").classList.remove("hidden");
  });

  $("review-button").addEventListener("click", writeReview);

  $("clear-button").addEventListener("click", () => {
    if (!state.movieId) return;
    state.histories[state.movieId] = [];
    saveHistories();
    renderChat();
    renderTalking();
  });
}

async function main() {
  bindEvents();
  renderChat();
  renderTalking();
  await Promise.all([loadStatus(), loadMovies().catch((e) => showError(`영화 목록을 불러오지 못했습니다. ${e.message}`))]);

  // /?movie=<영화 ID> 로 들어오면 그 영화의 대화를 바로 연다. (공유 링크)
  const linked = new URLSearchParams(location.search).get("movie");
  if (linked && state.labelById.has(linked)) selectMovie(linked);
}

main();
