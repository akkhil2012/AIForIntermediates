from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Few-shot examples + question
prompt = """
Q: What is the capital of France?
A: Paris.

Q: What is the capital of Japan?
A: Tokyo.

Q: What is the capital of Himachal Pradesh?
A:"""

response = generator(prompt, max_new_tokens=50)
print("Answer:", response[0]["generated_text"])



