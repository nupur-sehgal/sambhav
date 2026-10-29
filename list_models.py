import os, json, urllib.request
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise SystemExit("GOOGLE_API_KEY not found in .env")

url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=100"
req = urllib.request.Request(url, headers={"x-goog-api-key": key})
data = json.load(urllib.request.urlopen(req))

skip = ("image", "tts", "audio", "live", "embedding", "robotics", "veo", "imagen")
for m in data["models"]:
    name = m["name"].replace("models/", "")
    if "generateContent" in m.get("supportedGenerationMethods", []) and not any(s in name for s in skip):
        print(name)