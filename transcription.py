import assemblyai as aai

# Set your API key
aai.settings.api_key = "6113689a06a841d2a954cbe42e71143d"

# Transcribe your audio file
transcriber = aai.Transcriber()
transcript = transcriber.transcribe("D:/music/english/SATYAMVM1.ogg")

# Print each word with its timestamps (milliseconds)
for word in transcript.words:
    print(f"{word.text}: start={word.start}ms, end={word.end}ms")
