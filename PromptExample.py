
#requirement.txt:
#pip3 install transformers torch sentencepiece


from transformers import pipeline

# Load a simple text generation pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Ask a question
prompt = "Explain quantum computing in simple terms."
response = generator(prompt, max_new_tokens=200)

print(response[0]["generated_text"])  

  
