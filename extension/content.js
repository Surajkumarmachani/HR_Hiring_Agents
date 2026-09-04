/* Interview signals — Meet companion, interviewer side.
 *
 * WHAT THIS DOES
 * Injects the interviewer's working panel into a Google Meet tab: the
 * questions built from the CV, the answer score and counter-questions, and
 * the live read. It is a second client of the same HTTP API the web app uses;
 * no interview logic lives here, on purpose. A rule enforced in a content
 * script is a rule anybody can switch off with devtools.
 *
 * WHAT IT SENDS
 * Frames of the candidate's tile, sampled to a canvas and posted as JPEG on
 * /ws/meet/{sid}/{who}. The server authenticates the interviewer, checks the
 * CANDIDATE's consent, and refuses without it. The call is a different
 * transport, not a different legal basis.
 *
 * WHAT IT CANNOT FIX
 * The frames are whatever Meet's compositor produced: re-encoded, scaled to
 * the tile, and sampled at whatever rate this laptop manages while also
 * decoding a video call. So the achieved rate travels with every reading and
 * the pulse is withheld below a floor. See relay_quality in web/server.py.
 * A number that measured the network is worse than no number.
 */
(() => {
  "use strict";
  if (window.__ivs_loaded) return;
  window.__ivs_loaded = true;

  const CFG = { server: "", sid: "", who: "", token: "", myname: "" };
  const TARGET_FPS = 12;          // what we aim for; the server reports actual
  const FRAME_W = 480;            // wide enough for the ROI selector's patches
  const JPEG_Q = 0.72;

  let panel, cap, frameTimer, relayWs, readWs, canvas, ctx, tile = null;
  let state = null, capItems = [], capAt = 0;

  // ---------------------------------------------------------------- setup
  chrome.storage.local.get(["server", "link", "myname"], d => {
    CFG.myname = (d.myname || "").trim().toLowerCase();
    const link = d.link || "";
    // /i/{sid}/{who}?t={token} -- the same link the scheduler hands out, so
    // there is nothing extra for an interviewer to copy.
    const m = link.match(/\/i\/([^/]+)\/([^/?]+)\?t=([^&]+)/);
    if (!m) return note("Open the extension and paste your interviewer link.");
    CFG.server = (d.server || link.split("/i/")[0]).replace(/\/$/, "");
    CFG.sid = m[1]; CFG.who = m[2]; CFG.token = decodeURIComponent(m[3]);
    build();
    connectRead();
    watchForTile();
    watchCaptions();
  });

  const api = (path, opts) =>
    fetch(CFG.server + path, opts).then(r => r.json().then(j => ({ r, j })));

  // ------------------------------------------------------------- the panel
  function build() {
    panel = document.createElement("div");
    panel.id = "ivs";
    panel.innerHTML = `
      <header>
        <b>interview signals</b>
        <span id="ivs-state">connecting</span>
        <button id="ivs-cc" aria-pressed="true" title="captions">CC</button>
        <button id="ivs-min" title="collapse">&minus;</button>
      </header>
      <div id="ivs-body">
        <div class="ivs-sec" id="ivs-q"></div>
        <div class="ivs-sec" id="ivs-score"></div>
        <div class="ivs-sec" id="ivs-read"></div>
      </div>`;
    document.body.appendChild(panel);
    panel.querySelector("#ivs-min").onclick = () =>
      panel.classList.toggle("mini");
    const cc = panel.querySelector("#ivs-cc");
    cc.onclick = () => {
      const on = cc.getAttribute("aria-pressed") !== "true";
      cc.setAttribute("aria-pressed", on ? "true" : "false");
      drawCaption();
    };

    cap = document.createElement("div");
    cap.id = "ivs-cap";
    cap.hidden = true;
    document.body.appendChild(cap);
    refresh();
  }

  function note(msg) {
    const el = document.getElementById("ivs-state");
    if (el) el.textContent = msg;
    else console.warn("[interview signals]", msg);
  }

  // --------------------------------------------------- finding the candidate
  //
  // Meet gives no stable hook for "the other person's video". What it does
  // give is <video> elements with live MediaStreams, and the remote one is
  // reliably the largest that is not muted and not our own preview -- our own
  // is the one whose stream id matches a local capture, and simpler than
  // matching ids: our self-view is mirrored and small, the speaker tile is
  // large. Picking by area is crude and survives layout changes, which
  // matching class names does not.
  //
  // Re-picked continuously, because Meet swaps the element on every layout
  // change: pin, present, someone joins.
  function pickTile() {
    let best = null, bestArea = 0;
    for (const v of document.querySelectorAll("video")) {
      // A live remote stream, actually decoding, big enough to measure.
      if (!v.srcObject || v.readyState < 2) continue;
      if (v.videoWidth < 120 || v.paused) continue;
      const r = v.getBoundingClientRect();
      // Off-screen or collapsed elements: Meet keeps several around.
      if (r.width < 80 || r.height < 60) continue;
      const area = r.width * r.height;
      if (area > bestArea) { best = v; bestArea = area; }
    }
    return best;
  }

  function watchForTile() {
    setInterval(() => {
      const t = pickTile();
      if (t && t !== tile) {
        tile = t;
        note("candidate tile found");
        startRelay();
      } else if (!t && tile) {
        tile = null;
        stopRelay("no video tile in this call");
      }
    }, 1500);
  }

  // ------------------------------------------------------------- the relay
  function startRelay() {
    stopRelay();
    canvas = canvas || document.createElement("canvas");
    ctx = ctx || canvas.getContext("2d", { willReadFrequently: true });

    const url = CFG.server.replace(/^http/, "ws") +
      `/ws/meet/${CFG.sid}/${CFG.who}?t=${encodeURIComponent(CFG.token)}`;
    relayWs = new WebSocket(url);
    relayWs.onopen = () => {
      note("relaying");
      frameTimer = setInterval(sendFrame, Math.round(1000 / TARGET_FPS));
    };
    relayWs.onmessage = e => {
      // The server refuses with JSON before closing, so the reason is
      // reportable rather than a bare socket close code.
      try {
        const d = JSON.parse(e.data);
        if (d.error) note(d.reason || d.error);
      } catch (_) {}
    };
    relayWs.onclose = () => { clearInterval(frameTimer); frameTimer = null; };
  }

  function stopRelay(why) {
    clearInterval(frameTimer); frameTimer = null;
    if (relayWs) { try { relayWs.close(); } catch (_) {} relayWs = null; }
    if (why) note(why);
  }

  function sendFrame() {
    if (!tile || !relayWs || relayWs.readyState !== 1) return;
    if (tile.videoWidth < 80 || tile.paused) return;
    const w = FRAME_W;
    const h = Math.round(tile.videoHeight * (w / tile.videoWidth));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
    try {
      ctx.drawImage(tile, 0, 0, w, h);
    } catch (_) {
      // A tainted or not-yet-decoding element. Skip; the next tick retries.
      return;
    }
    canvas.toBlob(b => {
      if (b && relayWs && relayWs.readyState === 1) relayWs.send(b);
    }, "image/jpeg", JPEG_Q);
  }

  // ------------------------------------------------- reading back the panel
  function connectRead() {
    const url = CFG.server.replace(/^http/, "ws") +
      `/ws/interviewer/${CFG.sid}/${CFG.who}?t=${encodeURIComponent(CFG.token)}`;
    readWs = new WebSocket(url);
    readWs.onmessage = e => {
      let d; try { d = JSON.parse(e.data); } catch (_) { return; }
      if (d.type === "capture" || d.face || d.pulse || d.source) renderRead(d);
      if (d.line) { /* transcript lines arrive here too */ }
    };
    readWs.onclose = () => setTimeout(connectRead, 3000);
    setInterval(refresh, 4000);
  }

  async function refresh() {
    const { r, j } = await api(
      `/api/sessions/${CFG.sid}/interview?who=${CFG.who}` +
      `&t=${encodeURIComponent(CFG.token)}`);
    if (!r.ok) return note(j.detail || "cannot reach the server");
    state = j;
    renderQuestions();
    renderScore();
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g,
      c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function renderQuestions() {
    const asked = new Set(state.asked || []);
    const qs = (state.generated_questions || []).filter(q => !asked.has(q.id));
    const box = document.getElementById("ivs-q");
    if (!qs.length) {
      box.innerHTML = `<p class="ivs-hint">No unasked questions. Upload the CV
        in the web app to build them.</p>`;
      return;
    }
    const q = qs[0];
    box.innerHTML = `
      <h4>next question</h4>
      <p class="ivs-q">${esc(q.text)}</p>
      <p class="ivs-hint">${esc((q.competencies || []).join(", "))}</p>
      <button class="ivs-btn" id="ivs-ask">Ask this</button>`;
    box.querySelector("#ivs-ask").onclick = async () => {
      await api(`/api/sessions/${CFG.sid}/questions/asked`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ who: CFG.who, t: CFG.token, question_id: q.id }),
      });
      refresh();
    };
  }

  function renderScore() {
    const reads = state.assessments || [];
    const box = document.getElementById("ivs-score");
    const a = reads[reads.length - 1];
    if (!a || !a.read) { box.innerHTML = ""; capItems = []; drawCaption(); return; }
    const r = a.read;
    box.innerHTML = `
      <h4>last answer</h4>
      <p class="ivs-score">${r.score_out_of_10 == null ? "" :
        esc(r.score_out_of_10) + "<em>/10</em>"}
        <span>${esc(r.score_band || "")}</span></p>
      <p class="ivs-hint">${esc(r.score_reason || "")}</p>`;
    const ids = new Set(a.counter_question_ids || []);
    capItems = (state.suggestions || []).filter(s => ids.has(s.id));
    capAt = 0;
    drawCaption();
  }

  // The live read. `pulse_ok` false means the relay cannot support a pulse
  // estimate, and the tile says why instead of showing a number -- see
  // relay_quality in web/server.py.
  function renderRead(d) {
    const box = document.getElementById("ivs-read");
    if (d.type === "capture" && d.camera === false) {
      box.innerHTML = `<h4>live read</h4>
        <p class="ivs-hint">${esc(d.reason || "capture has stopped")}</p>`;
      return;
    }
    const rows = [];
    if (d.relay_fps != null) {
      rows.push(["capture", `${d.relay_fps} fps` +
        (d.relay_jitter_ms != null ? ` · ${d.relay_jitter_ms} ms jitter` : "")]);
    }
    const p = d.pulse || {};
    if (d.pulse_ok === false) {
      rows.push(["pulse", "not estimated"]);
    } else if (p.bpm != null) {
      rows.push(["pulse", `${Math.round(p.bpm)} bpm` +
        (p.roi_spread_bpm != null ? ` · regions differ ${
          Math.round(p.roi_spread_bpm)}` : "")]);
    }
    if (d.face && d.face.face_detected != null) {
      rows.push(["face", d.face.face_detected ? "in frame" : "not in frame"]);
    }
    box.innerHTML = `<h4>live read</h4>` +
      rows.map(([k, v]) =>
        `<p class="ivs-row"><span>${esc(k)}</span><b>${esc(v)}</b></p>`).join("") +
      (d.relay_note ? `<p class="ivs-warn">${esc(d.relay_note)}</p>` : "") +
      `<p class="ivs-hint">Descriptive. Not a score, and not an input to one.</p>`;
  }

  // ------------------------------------------------- reading Meet captions
  //
  // WHY CAPTIONS AND NOT THE TAB'S AUDIO
  // The audio would transcribe better -- it would go through the same
  // faster-whisper the direct path uses -- but the tab gives ONE MIXED
  // track, and splitting it back into speakers is diarisation, which this
  // project deliberately avoids by taking a separate track per participant.
  // Meet's captions are already attributed, because Google has the
  // per-participant streams. The attribution arrives correct by
  // construction, which is the property that matters: the whole
  // question/answer rule rests on knowing who spoke.
  //
  // WHY THIS IS THE FRAGILE PART
  // Meet exposes no API for its captions and no stable class names. Every
  // selector below will break on some future release, so they are tried in
  // order from most to least specific and the panel reports which one
  // worked -- a scraper that silently finds nothing looks exactly like a
  // candidate who is not talking.
  const CAP_ROOTS = [
    '[role="region"][aria-label*="aption" i]',
    'div[jsname="dsyhDe"]',
    'div[jsname="tgaKEf"]',
    '.a4cQT',
  ];
  const COMMIT_MS = 1400;     // no change for this long => the line is done
  const MAX_LINE = 400;

  let capRoot = null, capRootWhich = "";
  let pending = new Map();    // speaker -> {text, at}
  let outbox = [];

  function findCaptionRoot() {
    for (const sel of CAP_ROOTS) {
      const el = document.querySelector(sel);
      if (el) { capRootWhich = sel; return el; }
    }
    return null;
  }

  // Meet renders one block per speaker turn: a short name and the text.
  // Read STRUCTURALLY rather than by class -- the name is the short
  // punctuation-free string, the caption is the long one -- because that
  // shape has survived several Meet redesigns and class names have not.
  function readBlocks(root) {
    const out = [];
    for (const block of root.children) {
      const texts = [];
      const walk = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
      for (let n = walk.nextNode(); n; n = walk.nextNode()) {
        const t = (n.textContent || "").trim();
        if (t) texts.push(t);
      }
      if (!texts.length) continue;
      let name = "", said = "";
      for (const t of texts) {
        if (!name && t.length <= 40 && !/[.!?]$/.test(t) &&
            t.split(/\s+/).length <= 5) { name = t; continue; }
        said = said ? said + " " + t : t;
      }
      if (!said && texts.length === 1) { said = texts[0]; name = name || ""; }
      if (said) out.push({ name, text: said.slice(0, MAX_LINE) });
    }
    return out;
  }

  // The interviewer's own Meet display name decides the role. Asked for in
  // the popup rather than guessed: getting it wrong puts the candidate's
  // answer on the interviewer's ledger, and the answer window is built on
  // exactly that distinction. In a two-person call, "not me" is the
  // candidate, which is exact.
  function roleOf(name) {
    const n = (name || "").trim().toLowerCase();
    if (!n) return "candidate";
    if (CFG.myname && n.includes(CFG.myname)) return "interviewer";
    if (CFG.myname) return "candidate";
    return null;              // cannot tell: do not guess
  }

  function watchCaptions() {
    setInterval(() => {
      if (!capRoot || !capRoot.isConnected) {
        capRoot = findCaptionRoot();
        if (capRoot) note("captions found (" + capRootWhich + ")");
      }
      if (!capRoot) {
        note("turn Meet captions on (CC in the call toolbar)");
        return;
      }
      scanCaptions();
      commitIdle();
      flush();
    }, 500);
  }

  // Meet REVISES a caption in place as recognition improves: "So we" becomes
  // "So we will" becomes "So we will look at". Posting every mutation would
  // send a dozen partial duplicates of one sentence and the answer score
  // would be computed on a prefix. So a turn is held until it stops
  // growing, and only then committed.
  function scanCaptions() {
    for (const { name, text } of readBlocks(capRoot)) {
      const role = roleOf(name);
      if (!role) continue;
      const cur = pending.get(role);
      if (!cur) { pending.set(role, { text, at: Date.now() }); continue; }
      if (text === cur.text) continue;
      if (text.startsWith(cur.text) || cur.text.startsWith(text)) {
        // Same utterance, refined. Keep the longer reading.
        cur.text = text.length > cur.text.length ? text : cur.text;
        cur.at = Date.now();
      } else {
        // No longer an extension of what we had: the previous turn ended.
        queue(role, cur.text);
        pending.set(role, { text, at: Date.now() });
      }
    }
  }

  function commitIdle() {
    const now = Date.now();
    for (const [role, cur] of [...pending]) {
      if (now - cur.at >= COMMIT_MS) {
        queue(role, cur.text);
        pending.delete(role);
      }
    }
  }

  const sent = new Set();
  function queue(role, text) {
    text = (text || "").trim();
    if (!text) return;
    const key = role + "|" + text;
    if (sent.has(key)) return;      // Meet re-renders old blocks on scroll
    sent.add(key);
    if (sent.size > 400) sent.clear();
    outbox.push({ speaker: role === "interviewer" ? "interviewer" : "candidate",
                  text, t: Math.round((Date.now() - t0) / 100) / 10 });
  }

  const t0 = Date.now();
  let flushing = false;
  async function flush() {
    if (flushing || !outbox.length) return;
    flushing = true;
    const batch = outbox.splice(0, 20);
    try {
      const { r, j } = await api(
        `/api/sessions/${CFG.sid}/relay/transcript`, {
          method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ who: CFG.who, t: CFG.token, lines: batch }),
        });
      if (!r.ok) {
        // Consent refusals are terminal, not transient: stop trying and say
        // why, rather than retrying a 403 every half second forever.
        note(j.detail || "transcript refused");
        if (r.status === 403 || r.status === 409) outbox.length = 0;
      }
    } catch (_) {
      outbox.unshift(...batch);     // network blip: keep them for next tick
    } finally {
      flushing = false;
    }
  }

  // ------------------------------------------------------------- captions
  function drawCaption() {
    const on = document.getElementById("ivs-cc")
      ?.getAttribute("aria-pressed") === "true";
    if (!on || !capItems.length) { cap.hidden = true; return; }
    capAt = Math.max(0, Math.min(capAt, capItems.length - 1));
    const it = capItems[capAt];
    cap.hidden = false;
    cap.innerHTML = `
      <div class="ivs-cap-head">counter-question${
        capItems.length > 1 ? ` · ${capAt + 1}/${capItems.length}` : ""}
        ${capItems.length > 1
          ? `<button id="ivs-cap-prev">&lsaquo;</button>
             <button id="ivs-cap-next">&rsaquo;</button>` : ""}</div>
      <p class="ivs-cap-text">${esc(it.text)}</p>
      ${it.listen_for
        ? `<p class="ivs-cap-sub">${esc(it.listen_for)}</p>` : ""}`;
    const prev = cap.querySelector("#ivs-cap-prev");
    const next = cap.querySelector("#ivs-cap-next");
    if (prev) prev.onclick = () => { capAt--; drawCaption(); };
    if (next) next.onclick = () => { capAt++; drawCaption(); };
  }
})();
