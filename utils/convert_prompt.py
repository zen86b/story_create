from llama_cpp import Llama


def convert_prompt(inputs):
	
	llm = Llama(
		model_path="model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Download the model file first
		n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
		n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
		n_gpu_layers=35,  # The number of layers to offload to GPU, if you have GPU acceleration available
		chat_format="llama-2",
		verbose=False,
	)
	outputs = []
	for input in inputs:
		output = llm.create_chat_completion(
			messages=[
				{
					"role": "system",
					"content": "You are a story summary assistant. You need to turn what I say to a description (less than 60 words) of a picture that I can use to generate a hyper-realistic picture of a person, place, or thing.",
				},
				{
					"role": "user",
					"content": f"{input}",
				},
			]
		)
		print(output["choices"][0]["message"]["content"])
		outputs.append(output["choices"][0]["message"]["content"])
	return outputs


if __name__ == "__main__":
	convert_prompt(["In the year 2042, robots were no longer a novelty. They were everywhere, doing all sorts of jobs that humans used to do."])