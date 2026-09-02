#!/usr/bin/env python3
"""Check the question-generation credential, without starting the server.

    python3 check_api.py

Exists because the failure modes here are all configuration, they are
indistinguishable from each other in the UI, and discovering one mid-interview
is the worst possible time. This makes one cheap real request and says exactly
what is wrong and what to do about it.

Nothing here touches an interview, a candidate, or any recorded data.
"""
import os
import sys

import env_file


def main():
    displaced = env_file.displaced_placeholders()
    loaded = env_file.load()
    print()
    if loaded:
        print(f"  .env supplied: {', '.join(loaded)}")
    if displaced:
        print(f"  note: your shell has a placeholder for "
              f"{', '.join(displaced)}; .env was used instead")

    from config import CONFIG
    from interview.generate import _supplied_by
    supplied = _supplied_by()
    key = os.environ.get(supplied or "", "")
    print(f"  provider:               Google Gemini")
    print(f"  model:                  {CONFIG.generation.gemini_model}")
    print(f"  {(supplied or 'GEMINI_API_KEY') + ':':23s} "
          f"{'set (' + str(len(key)) + ' chars)' if key else 'NOT SET'}")
    print()

    if "--list-models" in sys.argv:
        # Availability differs by project, and the listing includes RETIRED
        # models -- it is not proof of callability, only a real request is.
        try:
            from google import genai
            client = genai.Client(api_key=key)
            print("  models this key lists (some may be retired):")
            for m_ in client.models.list():
                acts = getattr(m_, "supported_actions", None) or []
                if not acts or "generateContent" in acts:
                    print(f"    {m_.name}")
            return 0
        except Exception as e:
            print(f"  FAIL  could not list models: {type(e).__name__}: {e}")
            return 1

    if "--list-models" in sys.argv and provider == "gemini":
        # Availability differs by project and by key, so listing is the only
        # reliable way to choose a model id.
        try:
            from google import genai
            client = genai.Client(api_key=key)
            print("  models this key can call:")
            for m in client.models.list():
                methods = getattr(m, "supported_actions", None) or []
                if not methods or "generateContent" in methods:
                    print(f"    {m.name}")
            return 0
        except Exception as e:
            print(f"  FAIL  could not list models: {type(e).__name__}: {e}")
            return 1

    from interview.generate import _key_diagnosis
    bad = _key_diagnosis()
    if bad:
        print(f"  FAIL  {bad}")
        return 1
    if not key:
        print("  FAIL  No key. Put GEMINI_API_KEY in .env "
              "(see .env.example).")
        return 1

    # One real generation-shaped call through the same dispatcher the server
    # uses, so this tests the path that actually runs rather than a
    # near-neighbour of it.
    if True:
        from interview.generate import generate_json, GenerationError
        schema = {"type": "object",
                  "properties": {"ok": {"type": "boolean"}},
                  "required": ["ok"]}
        try:
            data, prov = generate_json(
                "Reply with JSON only.", 'Set ok to true.', schema)
        except GenerationError as e:
            print(f"  FAIL  {e}")
            return 1
        print(f"  PASS  credential works. Model replied {data!r}.")
        print(f"        served by {prov.get('served_by_model')}, "
              f"request {prov.get('request_id')}")
        print()
        print("  Resume-derived questions will work. Next:")
        print("    python3 tests/test_generate.py --live")
        print("    python3 run_web.py")
        return 0



if __name__ == "__main__":
    sys.exit(main())
