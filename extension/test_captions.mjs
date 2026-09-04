/* The caption dedupe logic, tested without a browser.
 *
 * Meet REVISES a caption in place as recognition improves, and re-renders old
 * blocks when the list scrolls. Both produce near-duplicates of one sentence.
 * If those reach the server, the answer score is computed on a prefix -- or
 * on the same sentence six times -- and neither failure is visible in the
 * output. So the logic is extracted here and driven with the real mutation
 * sequence rather than trusted.
 */
const COMMIT_MS = 1400;

function makeReader(myname) {
  const pending = new Map(), outbox = [], sent = new Set();
  let now = 0;
  const roleOf = n => {
    const s = (n || "").trim().toLowerCase();
    if (!s) return "candidate";
    if (myname && s.includes(myname)) return "interviewer";
    return myname ? "candidate" : null;
  };
  const queue = (role, text) => {
    text = (text || "").trim();
    if (!text) return;
    const key = role + "|" + text;
    if (sent.has(key)) return;
    sent.add(key);
    outbox.push({ speaker: role, text });
  };
  return {
    outbox,
    tick(blocks, advanceMs = 100) {
      now += advanceMs;
      for (const { name, text } of blocks) {
        const role = roleOf(name);
        if (!role) continue;
        const cur = pending.get(role);
        if (!cur) { pending.set(role, { text, at: now }); continue; }
        if (text === cur.text) continue;
        if (text.startsWith(cur.text) || cur.text.startsWith(text)) {
          cur.text = text.length > cur.text.length ? text : cur.text;
          cur.at = now;
        } else { queue(role, cur.text); pending.set(role, { text, at: now }); }
      }
      for (const [role, cur] of [...pending]) {
        if (now - cur.at >= COMMIT_MS) { queue(role, cur.text); pending.delete(role); }
      }
    },
  };
}

let fail = 0;
const check = (name, ok, detail = "") => {
  console.log(`   [${ok ? "ok  " : "FAIL"}] ${name}${detail ? "   " + detail : ""}`);
  if (!ok) fail++;
};

console.log("\n1. A caption refined in place commits once, in full");
let r = makeReader("alice");
r.tick([{ name: "Suraj", text: "So we" }]);
r.tick([{ name: "Suraj", text: "So we batched" }]);
r.tick([{ name: "Suraj", text: "So we batched the deletes" }]);
r.tick([{ name: "Suraj", text: "So we batched the deletes in ten thousand row chunks." }]);
check("nothing committed while it is still growing", r.outbox.length === 0);
r.tick([], 1500);
check("committed once when it stopped changing", r.outbox.length === 1,
      `${r.outbox.length} line(s)`);
check("and in its final, complete form",
      r.outbox[0].text.endsWith("ten thousand row chunks."), r.outbox[0].text);

console.log("\n2. A new utterance commits the previous one");
r = makeReader("alice");
r.tick([{ name: "Suraj", text: "First answer about latency" }]);
r.tick([{ name: "Suraj", text: "Completely different second thought" }]);
check("the first is flushed as soon as the second is not an extension of it",
      r.outbox.length === 1 && r.outbox[0].text === "First answer about latency",
      JSON.stringify(r.outbox.map(o => o.text)));

console.log("\n3. Re-rendered old blocks are not sent twice");
r = makeReader("alice");
r.tick([{ name: "Suraj", text: "A settled sentence." }]);
r.tick([], 1500);
check("committed once", r.outbox.length === 1);
r.tick([{ name: "Suraj", text: "A settled sentence." }]);
r.tick([], 1500);
check("the same text re-appearing on scroll is ignored", r.outbox.length === 1,
      `${r.outbox.length} line(s)`);

console.log("\n4. Roles come from the interviewer's own display name");
r = makeReader("alice");
r.tick([{ name: "Alice Kumar", text: "Tell me about the retention work" }]);
r.tick([{ name: "Suraj", text: "We batched the deletes" }]);
r.tick([], 1500);
const roles = r.outbox.map(o => o.speaker);
check("the interviewer's own lines are attributed to the interviewer",
      roles.includes("interviewer"), JSON.stringify(roles));
check("everyone else is the candidate", roles.includes("candidate"));
check("both speakers are kept apart", new Set(roles).size === 2);

console.log("\n5. Without a display name it refuses to guess");
r = makeReader("");
r.tick([{ name: "Someone", text: "Words that could be either side" }]);
r.tick([], 1500);
check("no line is attributed at all", r.outbox.length === 0,
      "misattributing an answer to the interviewer would corrupt the "
      + "answer window");

console.log();
if (fail) { console.log(`FAIL — ${fail} check(s)`); process.exit(1); }
console.log("PASS — captions commit once, in full, attributed correctly.");
