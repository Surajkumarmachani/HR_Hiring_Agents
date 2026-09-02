"""The interviewer's camera: relayed to the candidate, and never measured.

Run:  python3 tests/test_presence.py

The candidate is the data subject; the interviewer is staff. That asymmetry is
the whole design of this path, and it is the thing worth asserting, because
the failure modes are silent in both directions:

  - if a presence frame ever reached the analyzer, the interviewer would be
    getting measured -- unconsented, and mixed into the candidate's own
    session state, which would corrupt the measurement as well as the ethics;
  - if two panel members could publish at once, the candidate's single feed
    would interleave two JPEG streams and show a flicker, and the second
    interviewer would have no way to know why.

Neither shows up as an exception, so both are checked here. This exercises the
Hub directly rather than through a server, so it needs no web framework and no
sockets -- and the endpoints in web/server.py are a thin wrapper over exactly
these calls.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.live import Hub

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class FakeSocket:
    """Records what was sent. `broken` refuses, like a closed browser tab."""

    def __init__(self, broken=False):
        self.frames = []
        self.states = []
        self.broken = broken

    async def send_bytes(self, b):
        if self.broken:
            raise ConnectionError("closed")
        self.frames.append(b)

    async def send_json(self, d):
        if self.broken:
            raise ConnectionError("closed")
        self.states.append(d)


print("\n1. One camera slot, held by whoever claimed it first")
hub = Hub()
check("the first panel member gets the slot", hub.claim_camera("s1", "ravi"))
check("a second is refused", not hub.claim_camera("s1", "meera"))
check("the holder is unchanged by the refusal",
      hub.on_camera["s1"] == "ravi", hub.on_camera["s1"])
check("the holder can reclaim its own slot (a reconnect, not a conflict)",
      hub.claim_camera("s1", "ravi"))
check("another session is independent", hub.claim_camera("s2", "meera"))

print("\n2. The slot is released only by the person holding it")
hub.release_camera("s1", "meera")
check("a non-holder cannot release it", hub.on_camera.get("s1") == "ravi")
hub.release_camera("s1", "ravi")
check("the holder can", hub.on_camera.get("s1") is None)
check("and the slot is then free", hub.claim_camera("s1", "meera"))

print("\n3. State says who is on camera, so the candidate can be told")
hub = Hub()
check("nobody on camera reports None", hub.camera_state("s1") ==
      {"type": "presence", "on_camera": None})
hub.claim_camera("s1", "ravi")
check("a claim names them", hub.camera_state("s1") ==
      {"type": "presence", "on_camera": "ravi"})

print("\n4. Frames reach every viewer, and a dead socket is dropped")
hub = Hub()
good, also, dead = FakeSocket(), FakeSocket(), FakeSocket(broken=True)
for ws in (good, also, dead):
    hub.viewers["s1"].add(ws)
asyncio.run(hub.send_to_candidate("s1", frame=b"\xff\xd8jpeg"))
check("every live viewer got the frame",
      good.frames == [b"\xff\xd8jpeg"] and also.frames == [b"\xff\xd8jpeg"])
check("the dead socket was discarded rather than retried forever",
      dead not in hub.viewers["s1"] and len(hub.viewers["s1"]) == 2)
asyncio.run(hub.send_to_candidate("s1", state=hub.camera_state("s1")))
check("state fans out on the same path", good.states[-1]["type"] == "presence")
check("a session with no viewers is not an error",
      asyncio.run(hub.send_to_candidate("s9", frame=b"x")) is None)

print("\n5. Nothing on this path is measured, transcribed or kept")
hub = Hub()
hub.claim_camera("s1", "ravi")
hub.viewers["s1"].add(FakeSocket())
for _ in range(30):
    asyncio.run(hub.send_to_candidate("s1", frame=b"\xff\xd8jpeg"))
# The measured path builds these lazily, so "empty" is the proof that a
# presence frame never entered it. If a future change routes these frames
# through the analyzer for a thumbnail or a preview, this is what catches it.
check("no analyzer was created for the interviewer's frames",
      hub.analyzers == {}, repr(hub.analyzers))
check("no transcriber was created", hub.transcribers == {})
check("no measurement snapshot was retained", hub.latest == {})
check("no session clock was started", hub.started == {})
check("the relay keeps no frame buffer of its own",
      not any(isinstance(v, (bytes, bytearray))
              for v in vars(hub).values()))

print("\n6. The publisher is told how many people can see them")
# An interviewer whose camera is on but whose candidate has not joined is
# visible to nobody. The page must be able to say so, which means the count
# has to come from the server -- the browser only knows its own camera is on.
hub = Hub()
pub = FakeSocket()
hub.claim_camera("s1", "ravi", pub)
asyncio.run(hub.tell_publisher("s1"))
check("nobody watching reports zero", pub.states[-1] == {"type": "viewers", "count": 0},
      str(pub.states[-1]))
hub.viewers["s1"].add(FakeSocket())
asyncio.run(hub.tell_publisher("s1"))
check("a joined candidate reports one", pub.states[-1] == {"type": "viewers", "count": 1},
      str(pub.states[-1]))
hub.release_camera("s1", "ravi")
asyncio.run(hub.tell_publisher("s1"))
check("no publisher is not an error", pub.states[-1]["count"] == 1)
check("releasing the slot forgets the socket", "s1" not in hub.publishers)
hub.claim_camera("s2", "meera", FakeSocket(broken=True))
asyncio.run(hub.tell_publisher("s2"))
check("a dead publisher socket is dropped, not retried",
      "s2" not in hub.publishers)

print("\n7. The relay is separate from the measured fan-out")
# Watchers receive frames that WILL be measured; viewers receive frames that
# never are. Crossing the two would send the candidate their own face, and
# the interviewer's face into the measurement panel.
hub = Hub()
watcher, viewer = FakeSocket(), FakeSocket()
hub.watchers["s1"].add(watcher)
hub.viewers["s1"].add(viewer)
asyncio.run(hub.send_to_candidate("s1", frame=b"interviewer"))
asyncio.run(hub.broadcast("s1", frame=b"candidate"))
check("the interviewer's frame reached only the candidate",
      viewer.frames == [b"interviewer"])
check("the candidate's frame reached only the panel",
      watcher.frames == [b"candidate"])

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — the interviewer's camera reaches the candidate and nothing else.")
