from jiwer import wer, cer

with open("original.txt", "r") as f:
    reference = f.read().strip()

with open("transcription.txt", "r") as f:
    hypothesis = f.read().strip()

print("WER:", wer(reference, hypothesis))
print("CER:", cer(reference, hypothesis))
print("Accuracy:", 1 - wer(reference, hypothesis))
