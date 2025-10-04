from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import BartForConditionalGeneration, BartTokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[ logging.StreamHandler() ]
)
logger = logging.getLogger(__name__)

def lexrank_summary(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    logger.info("Extractive summary completed successfully.")
    return " ".join([str(sentence) for sentence in summary])

def bart_summary(text, max_length=500):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_length,
        min_length=100,
        no_repeat_ngram_size=3
    )

    logger.info("Abstractive summary completed successfully.")
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    with open("transcription.txt", "r") as f:
        transcript = f.read()
    summary = lexrank_summary(transcript, num_sentences=5)
    # summary = bart_summary(transcript)
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)