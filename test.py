with open('transcription.txt', 'r', encoding='utf-8') as f:
    transcript = f.read()
    print(len(transcript))

with open('summary.txt', 'r', encoding='utf-8') as f:
    summary = f.read()
    print(len(summary))