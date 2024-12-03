from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn

def get_tokenizer_and_embs(model_name='bert-base-uncased'):
    tk = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)

    return tk, mdl

def embed(text, tk, mdl):
    inputter = mdl.get_input_embeddings()
    toks = torch.Tensor(tk(text)['input_ids']).int()

    return inputter(toks)

class CharModel(nn.Module):

if __name__ == "__main__":
    tk, mdl = get_tokenizer_and_embs()
    vec = embed('wombats are not technically human. technically.', tk, mdl)
    import pdb; pdb.set_trace()
