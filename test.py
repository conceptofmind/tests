from transformers import GPT2Tokenizer, OPTForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

model = OPTForCausalLM.from_pretrained("conceptofmind/code-350-model")

prompt = "def add(a, b): return"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=128)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])