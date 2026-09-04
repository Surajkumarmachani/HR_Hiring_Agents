const $ = s => document.querySelector(s);
chrome.storage.local.get(["server", "link", "myname"], d => {
  $("#server").value = d.server || "http://127.0.0.1:8000";
  $("#link").value = d.link || "";
  $("#myname").value = d.myname || "";
});
$("#save").onclick = () => {
  chrome.storage.local.set({
    server: $("#server").value.trim().replace(/\/$/, ""),
    link: $("#link").value.trim(),
    myname: $("#myname").value.trim(),
  }, () => window.close());
};
