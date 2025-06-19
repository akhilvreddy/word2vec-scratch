from datasets import load_dataset
import re

def clean_line(line):
    line = line.strip()
    if len(line) == 0 or line.startswith("|"):
        return None
    line = line.lower()
    line = re.sub(r'[^a-z\s]', '', line)
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def save_cleaned_text():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    lines = dataset["train"]["text"]

    cleaned_lines = [clean_line(line) for line in lines]
    cleaned_lines = [line for line in cleaned_lines if line]

    joined = " ".join(cleaned_lines)

    with open("data/cleaned.txt", "w") as f:
        f.write(joined)

    print("Cleaned and saved corpus. Total words:", len(joined.split()))

if __name__ == "__main__":
    save_cleaned_text()