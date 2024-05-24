from transformers import pipeline, Conversation

# text_list = ["This is great", "Thanks for nothing", "You've got to work on your face", "You're beautiful, never change!"]


# classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# print(classifier("Hate this"))

# classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions")
# print(classifier("let's go to for a tour in the mountains please, I am tired of this city life."))


chatbot = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")
conversation = Conversation('"Hi I am Asim, How are you?"')
conversation = chatbot(conversation)
print(conversation)
