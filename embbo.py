from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import lightning as L

def get_tokenizer_and_embs(model_name='bert-base-uncased'):
    tk = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)

    return tk, mdl

def embed(text, tk, mdl):
    inputter = mdl.get_input_embeddings()
    toks = torch.Tensor(tk(text)['input_ids']).int()

    return inputter(toks)

class NeuralTokenizer(L.Module):
    def __init__(self, d_model=256,
                 target_vocab=30522, # for bert
                 nhead=4,
                 num_encoder_layers=3,
                 max_output_length=2048):
        super()).__init__()
        self.byte_embeddings = nn.Embedding(256, d_model)
        self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_encoder_layers,
                batch_first=True,
                norm_first=True)

        self.token_predictor = nn.Linear(d_model, target_vocab)
        self.max_output_length = max_output_length

    def training_step(model, batch, batch_idx):
        x, y = batch
        x =x.view(x.size(0))
        y_hat = self.token_predictor(self.transformer(x))
        loss = nn.cross_entropy_loss(y, y_hat)
        self.log("train_loss", loss)
        return loss

if __name__ == "__main__":
    tk, mdl = get_tokenizer_and_embs()
    vec = embed('wombats are not technically human. technically.', tk, mdl)
    l, dim = vec.shape
    print(f"Target dim {dim}")
    nt = NeuralTokenizer(d_model=dim)
    import pdb; pdb.set_trace()

