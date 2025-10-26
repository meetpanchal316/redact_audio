import assemblyai as aai
import re
from pydub import AudioSegment
from pydub.generators import Sine
from collections import Counter
from gliner import GLiNER  # Using the new PII detection model

# ============ CONFIGURATION ============
INPUT_AUDIO_PATH = "D:/music/english/SATYAMVM1.ogg"
OUTPUT_AUDIO_PATH = "D:/music/english/censored_SATYAMVM1.mp3"

banned_words = {
    "fuck", "shit", "bitch", "asshole", "nigger", "slut", "dick",
    "bastard", "idiot", "stupid", "fag", "damn", "ass",
    "fucking", "stupidbitch", "fuckingasshole"
}
USE_BEEP = True
OFFSET_MS = 500

# ============ STEP 1: TRANSCRIBE AUDIO & ALIGN WORDS ============
API_KEY = "6113689a06a841d2a954cbe42e71143d"

aai.settings.api_key = API_KEY
transcriber = aai.Transcriber()
transcript = transcriber.transcribe(INPUT_AUDIO_PATH)

word_segments = []
all_words = []
for word in transcript.words:
    word_segments.append({
        "word": word.text,
        "start": word.start / 1000,  # AssemblyAI gives ms, convert to seconds
        "end": word.end / 1000
    })
    all_words.append(word.text)
transcript_text = " ".join(all_words)
print(" ".join(all_words))

# ============ STEP 3: PII DETECTION USING GLiNER ============
print("\n🔍 Running GLiNER PII detection...\n")
gliner_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

# Get detected entities
labels = ["PERSON", "ORG", "LOCATION", "EMAIL", "DATE", "PHONE", "IP", "CARDINAL"]
entities = gliner_model.predict_entities(transcript_text, labels)

# Flatten detected entities into a set of lowercase words for censoring
entities_to_censor = set()
for ent in entities:
    for word in ent["text"].lower().split():
        entities_to_censor.add(word)

# ============ STEP 4: LOAD AUDIO ============
print("\n🎧 Loading original audio...")
audio = AudioSegment.from_file(INPUT_AUDIO_PATH)

# ============ STEP 5: PREPARE BEEP SOUND ============
beep = Sine(1000).to_audio_segment(duration=300).apply_gain(-5)

# ============ STEP 6: FIND WORDS TO CENSOR ============
print("\n🚫 Checking and censoring sensitive words...\n")
word_intervals = []
censored_counter = Counter()

for word_info in word_segments:
    raw_word = word_info["word"]
    cleaned = re.sub(r"[^a-zA-Z0-9']+", "", raw_word.lower().strip())
    if not cleaned:
        continue
    if cleaned in banned_words or cleaned in entities_to_censor:
        start_ms = max(0, int(word_info["start"] * 1000) - OFFSET_MS)
        end_ms = int(word_info["end"] * 1000) - OFFSET_MS + 100
        word_intervals.append((start_ms, end_ms, cleaned))
        censored_counter[cleaned] += 1

# Merge overlapping/closely spaced intervals
merged_intervals = []
if word_intervals:
    word_intervals.sort(key=lambda x: x[0])
    current_start, current_end, merged_words = word_intervals[0][0], word_intervals[0][1], [word_intervals[0][2]]

    for start, end, word in word_intervals[1:]:
        if start <= current_end + 200:
            current_end = max(current_end, end)
            merged_words.append(word)
        else:
            merged_intervals.append((current_start, current_end, merged_words))
            current_start, current_end, merged_words = start, end, [word]
    merged_intervals.append((current_start, current_end, merged_words))

# ============ STEP 7: CENSOR AUDIO ============
if merged_intervals:
    for start_ms, end_ms, words in merged_intervals:
        duration = max(150, end_ms - start_ms)
        censored = beep[:duration] if USE_BEEP else AudioSegment.silent(duration=duration)
        censored = censored.fade_in(10).fade_out(10)
        audio = audio[:start_ms] + censored + audio[end_ms:]
        print(f"🚫 Censored words {words} ({start_ms/1000:.2f}s - {end_ms/1000:.2f}s)")
    found_any = True
else:
    found_any = False

# ============ STEP 8: EXPORT FINAL AUDIO & SUMMARY ============
audio.export(OUTPUT_AUDIO_PATH, format="mp3")

print("\n" + "=" * 40)
if found_any:
    total_censored = sum(censored_counter.values())
    print("✅ Censorship Summary:")
    print(f"Total censored words: {total_censored}")
    for word, count in censored_counter.items():
        print(f"  - {word}: {count} time(s)")
else:
    print("🎉 No banned words found in the audio!")
print(f"🔈 Censored audio saved at: {OUTPUT_AUDIO_PATH}")
print("=" * 40)
