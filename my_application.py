import gradio as gr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, Conversation, pipeline

chatbot = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

# Initialize the tokenizer and model
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Function to get response from the model
# def chat_with_bot(input_text, history=[]):
#     inputs = tokenizer([input_text], return_tensors='pt')
#     reply_ids = model.generate(**inputs)
#     response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
#     history.append((input_text, response))
#     return history, history

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    conversation = Conversation(message, system_message)
    # messages = [{"role": "system", "content": system_message}]
    messages = conversation.messages
    response = chatbot(message)

    # for val in history:
    #     if val[0]:
    #         messages.append({"role": "user", "content": val[0]})
    #     if val[1]:
    #         messages.append({"role": "assistant", "content": val[1]})

    # messages.append({"role": "user", "content": message})


    # for message in chat_completion(
    #     messages,
    #     max_tokens=max_tokens,
    #     stream=True,
    #     temperature=temperature,
    #     top_p=top_p,
    # ):
    #     token = message.choices[0].delta.content

        # response += token
    return response

# Create Gradio interface
# chatIFace = gr.Interface(
#     fn=chat_with_bot,
#     inputs=[gr.Textbox(lines=2, placeholder="Enter your message here..."), "state"],
#     outputs=[gr.Chatbot(), "state"],
#     live=False
# )

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

# Launch the interface
demo.launch()