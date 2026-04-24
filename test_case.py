import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/content/drive/MyDrive/Transformer/sentiment_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()


def preprocessing(text):
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()

def predict_sentiment(text):
    text = preprocessing(text)

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        probs = torch.softmax(logits, dim=1)[0]

    label_map = {
        0: "Negative",
        1: "Positive"
    }

    return {
        "text": text,
        "prediction": label_map[pred],
        "confidence": probs[pred].item()
    }

test_cases = [
    {"text": "I can't say I didn't enjoy it.", "label": 1},
    {"text": "It's not impossible to like this.", "label": 1},
    {"text": "I don't think it's not bad.", "label": 1},
    {"text": "I wouldn't say it was good.", "label": 0},

    # =========================
    # SARCASM
    # =========================
    {"text": "Great job, now the app won't even open.", "label": 0},
    {"text": "Fantastic, another update that breaks everything.", "label": 0},
    {"text": "Lovely, my order arrived broken again.", "label": 0},
    {"text": "Amazing service, they ignored me for a week.", "label": 0},

    # =========================
    # CONTRAST SHIFT
    # =========================
    {"text": "The design is beautiful, but everything else is terrible.", "label": 0},
    {"text": "The beginning was boring, but the ending was amazing.", "label": 1},
    {"text": "It sounds promising, but it fails miserably.", "label": 0},
    {"text": "It started badly, but turned out wonderful.", "label": 1},

    # =========================
    # MIXED SENTIMENT
    # =========================
    {"text": "I love the features, but I hate the performance.", "label": 0},
    {"text": "The performance is terrible, but I still love it.", "label": 1},
    {"text": "The food was awful, but the dessert saved the night.", "label": 1},
    {"text": "The support was helpful, but the product is unusable.", "label": 0},

    # =========================
    # IMPLIED SENTIMENT
    # =========================
    {"text": "I expected better.", "label": 0},
    {"text": "I've had worse.", "label": 1},
    {"text": "That could have gone better.", "label": 0},
    {"text": "It wasn't exactly a pleasant experience.", "label": 0},

    # =========================
    # POSITIVE WORDS BUT NEGATIVE MEANING
    # =========================
    {"text": "The app is insanely good at crashing.", "label": 0},
    {"text": "This product is perfect for wasting money.", "label": 0},
    {"text": "Excellent, it broke instantly.", "label": 0},

    # =========================
    # NEGATIVE WORDS BUT POSITIVE MEANING
    # =========================
    {"text": "This movie was wicked good.", "label": 1},
    {"text": "That performance was insanely good.", "label": 1},
    {"text": "This cake is ridiculously delicious.", "label": 1},

    # =========================
    # LONG CONTEXT REVERSAL
    # =========================
    {"text": "Although I was frustrated at first, after using it for a while I ended up loving it.", "label": 1},
    {"text": "Even though the first impression was great, after a few days it became unusable.", "label": 0},
    {"text": "At first I hated the interface, but now I can't live without it.", "label": 1},
    {"text": "I thought I would love it, but it ended up being a complete waste.", "label": 0},
]
label_map = {0: "Negative", 1: "Positive"}

correct = 0

for case in test_cases:
    case["label"] = label_map[case["label"]]

    pred = predict_sentiment(case["text"])["prediction"]
    print(f"Text: {case['text']}")
    print(f"Prediction: {pred}")
    print(f"Label: {case['label']}")
    print("\n")
    if pred == case["label"]:
        correct += 1

print(f"{correct} / {len(test_cases)}")
