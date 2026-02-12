import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Model loaded successfully!\n")

    def analyze(self, text):
        # 1. Preprocessing (Tokenization)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # 2. Model Inference
        with torch.no_grad(): # Disable gradient calculation for faster inference
            outputs = self.model(**inputs)

        # 3. Postprocessing (Logits -> Probabilities)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the label with the highest score
        predicted_class_id = logits.argmax().item()
        label = self.model.config.id2label[predicted_class_id]
        score = probabilities[0][predicted_class_id].item()

        return label, score

def main():
    print("   EmoSense: AI Sentiment Analyzer      \n")
    
    analyzer = SentimentAnalyzer()

    print("Type a sentence to analyze (or 'q' to quit):")
    
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
        
        if not user_input.strip():
            continue

        label, score = analyzer.analyze(user_input)
        
        # Visual output
        icon = "🟢" if label == "POSITIVE" else "🔴"
        print(f"{icon} Sentiment: {label} ({score:.2%})")

if __name__ == "__main__":
    main()