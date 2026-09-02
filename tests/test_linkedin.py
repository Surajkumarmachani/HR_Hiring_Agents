"""LinkedIn export import: the allowlist is the whole feature.

Run:  python3 tests/test_linkedin.py

A LinkedIn data export is not a CV. In the same folder as Positions.csv sit
the member's entire contact network, both sides of every private message,
their home address, birth date, and an ad-targeting file with inferred income
and politics. The person handing over the ZIP is thinking "here is my CV".

So the only thing worth testing hard is that nothing outside the professional
allowlist reaches the stored record -- including columns inside files that ARE
read, since Profile.csv carries headline and home address in the same row.
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import consent as consent_mod
import linkedin_archive as la

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# A realistic export: the professional files, and the ones that must not be read.
FILES = {
    "Profile.csv": (
        "First Name,Last Name,Address,Birth Date,Headline,Summary,Industry\n"
        "Suraj,Kumar,\"221B Baker St\",1995-04-02,Engineer,"
        "\"Ten years in signal processing.\",Software Development\n"),
    "Positions.csv": (
        "Company Name,Title,Description,Location,Started On,Finished On\n"
        "Machani,Lead Engineer,\"Built the signal layer.\",Bengaluru,Jan 2025,\n"),
    "Education.csv": (
        "School Name,Start Date,End Date,Notes,Degree Name,Activities\n"
        "IIT Patna,2013,2017,,B.Tech,Robotics\n"),
    "Skills.csv": "Name\nPython\nFastAPI\n",
    "Connections.csv": (
        "First Name,Last Name,Email Address,Company,Position,Connected On\n"
        "Priya,Raman,priya@example.com,Acme,CTO,01 Jan 2024\n"),
    "messages.csv": (
        "CONVERSATION ID,FROM,TO,DATE,CONTENT\n"
        "1,Suraj,Priya,2024-01-01,\"CONFIDENTIALDM\"\n"),
    "Ad_Targeting.csv": "Member Age,Income Bracket,Interests\n30-34,High,politics\n",
    "PhoneNumbers.csv": "Extension,Number,Type\n,+919999999999,Mobile\n",
    "Some_New_File.csv": "A,B\n1,2\n",      # LinkedIn adds files over time
}

root = tempfile.mkdtemp(prefix="li-test-")
src = os.path.join(root, "export")
os.makedirs(src)
for name, text in FILES.items():
    with open(os.path.join(src, name), "w") as fh:
        fh.write(text)

try:
    # ================================================================= 1
    print("1. Parsing: read the professional files, skip the rest")

    data, report = la.parse(src)
    read = {r["file"] for r in report["read"]}
    check("reads Positions", "Positions.csv" in read)
    check("reads Education", "Education.csv" in read)
    check("reads Skills", "Skills.csv" in read)
    check("reads Profile", "Profile.csv" in read)

    skipped = {r["file"] for r in report["skipped"]}
    for name in ("Connections.csv", "messages.csv", "Ad_Targeting.csv",
                 "PhoneNumbers.csv"):
        check(f"skips {name}", name in skipped)
    check("every skip states a reason",
          all(r["reason"] for r in report["skipped"]))

    # A file on neither list defaults to NOT read, and is named so somebody
    # notices it appeared.
    check("an unrecognised file is not read",
          "Some_New_File.csv" in report["unrecognised"])
    check("an unrecognised file produced no data",
          "some_new_file" not in data)

    # ================================================================= 2
    print("\n2. Column allowlist: Profile.csv carries both kinds of field")

    prof = data["profile"][0]
    check("keeps the headline", prof.get("Headline") == "Engineer")
    check("drops the home address", "Address" not in prof)
    check("drops the birth date", "Birth Date" not in prof)

    # ================================================================= 3
    print("\n3. Nothing sensitive survives into the stored record")

    consent_mod.create(
        "suraj", root, purpose="test", context="self",
        signals=["video_facial_features", "face_identity_template"],
        retention_days=7, data_fiduciary="t", withdrawal_contact="t",
        grievance_contact="t")
    out, rep = la.import_archive("suraj", src, root=root)
    check("writes the subset", os.path.exists(out))

    stored = json.load(open(out))
    blob = json.dumps(stored["profile"])
    for term, what in [("Baker St", "home address"), ("1995-04-02", "birth date"),
                       ("priya@example.com", "a connection's email"),
                       ("Priya", "a connection's name"),
                       ("CONFIDENTIALDM", "a private message"),
                       ("+919999999999", "a phone number"),
                       ("30-34", "age bracket"), ("politics", "inferred politics")]:
        check(f"stored profile has no {what}", term not in blob)

    # The report names skipped FILES and reasons. It must never carry a row.
    rblob = json.dumps(stored["import_report"])
    check("the report carries no row content",
          "Priya" not in rblob and "CONFIDENTIALDM" not in rblob
          and "Baker" not in rblob)

    check("the record says where it came from",
          "export" in stored["_note"] and "not scraped" in stored["_note"].lower())
    check("carries the unverified advisory",
          "unverified" in stored["profile"]["_advisory"].lower())

    # ================================================================= 4
    print("\n4. A ZIP is handled the same as a folder")

    zpath = os.path.join(root, "export.zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for name, text in FILES.items():
            z.writestr(f"Basic_LinkedInDataExport/{name}", text)
    zdata, zreport = la.parse(zpath)
    check("reads a ZIP", "positions" in zdata)
    check("same allowlist applies inside a ZIP",
          "Connections.csv" in {r["file"] for r in zreport["skipped"]})

    # ================================================================= 5
    print("\n5. Consent gating and erasure")

    consent_mod.withdraw("suraj", root=root, reason="test")
    check("withdrawal erases the LinkedIn subset",
          not os.path.exists(la.output_path("suraj", root)))
    try:
        la.import_archive("suraj", src, root=root)
        check("refuses import for a withdrawn subject", False, "imported")
    except Exception as e:
        check("refuses import for a withdrawn subject",
              "consent" in str(e).lower())

finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — professional subset only; network, messages, contact details "
      "and ad-targeting never leave the export")
