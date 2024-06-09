from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

summarizer = None
paraphraser_model = None
paraphraser_tokenizer = None

def load_models():
    global summarizer, paraphraser_model, paraphraser_tokenizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    if paraphraser_model is None or paraphraser_tokenizer is None:
        paraphraser_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
        paraphraser_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")

def summarize_text(text: str):
    if summarizer is None:
        load_models()
    summary = summarizer(text, max_length=1000, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def paraphrase_text(text: str):
    if paraphraser_model is None or paraphraser_tokenizer is None:
        load_models()
    batch = paraphraser_tokenizer(text, truncation=True, padding='longest', return_tensors="pt")
    translated = paraphraser_model.generate(**batch)
    paraphrased_text = paraphraser_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return paraphrased_text
