# EmoSense: AI Sentiment Analysis CLI

A simple yet powerful command-line interface (CLI) tool that uses the **DistilBERT** transformer model to classify the sentiment of text input as either Positive or Negative.

## 🚀 How It Works
1. **Tokenization:** Converts raw text into input IDs using `AutoTokenizer`.
2. **Inference:** Passes inputs through a fine-tuned `DistilBERT` model.
3. **Postprocessing:** Applies `SoftMax` to logits to generate confidence scores.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/EmoSense.git](https://github.com/YOUR_USERNAME/EmoSense.git)