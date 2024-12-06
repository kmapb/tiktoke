import sys
import torch
import neural_tokenizer as NT

from transformers import PreTrainedModel, GenerationConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn as nn

class EmbedReplacer:
    def __init__(self, model_name, embedding_model):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.embedding_model = embedding_model

    def forward(self, s: str , attention_mask=None, **kwargs):
        inputs_embeds = self.embedding_model.raw_forward(s)
        return self.model.forward(inputs_embeds, attention_mask, **kwargs)

class CustomEmbeddingModel(nn.Module):
    transformer: AutoModelForCausalLM
    embedding: NT.NeuralTokenizerModule
    def __init__(self, model_name, embedding_model):
        super().__init__(config=AutoConfig.from_pretrained(model_name))
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name)
        self.embedding = embedding_model
        
    def forward(self, input_bytes, attention_mask=None, **kwargs):
        # Your custom embedding process
        inputs_embeds = self.embedding(input_bytes)
        
        # Forward through the rest of the model
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs

    def generate(self, tokenizer, s: str, **kwargs):
        inputs_embeds = self.embedding.embed_text(s)
        return self.transformer.generate(inputs_embeds=inputs_embeds, **kwargs)

def decode_response(tokenizer, outputs):
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chat_session(model: AutoModelForCausalLM, tokenizer, device="cuda"):
    model = model.to(device)
    generation_args = GenerationConfig(max_new_tokens=1000, max_beams=5, num_return_sequences=1)
    prompt = "1+1="
    with torch.no_grad():
        print(f"User: {prompt}")
        outputs = model.generate(tokenizer, prompt, generation_config=generation_args)
        response = decode_response(tokenizer, outputs)
        print(f"Response: {response}")
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        with torch.no_grad():
            outputs = model.generate(tokenizer, user_input, generation_config=generation_args)
            response = decode_response(tokenizer, outputs)
            print(f"Response: {response}")
if __name__ == "__main__":
    embedder = NT.NeuralTokenizerModule.load_from_checkpoint(sys.argv[1])
    model_name = NT.MODEL
    tform = CustomEmbeddingModel(model_name, embedder)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat_session(tform, tokenizer)
