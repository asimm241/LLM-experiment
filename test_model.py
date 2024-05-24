import prepare_and_tokenize
import torch
text_list = ["It was good", "Not a fan",
             "don't recommend", "Better than the first one", "This is not worth watching once", "This one is a pass"]

print("Untrained model predictions")
print("-----------------------------------------")
for text in text_list:
    inputs = prepare_and_tokenize.tokenizer.encode(text, return_tensors="pt")
    logits = prepare_and_tokenize.model(inputs).logits
    predictions = torch.argmax(logits)

    print(text + " - " + prepare_and_tokenize.id2label[predictions.tolist()])
