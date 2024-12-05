import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

MAX_SIZE=512
TXF_HEADS=4
TXF_LAYERS=1
MODEL='Qwen/Qwen2.5-Coder-0.5B-Instruct'

class WikipediaByteDataset(Dataset):
    def __init__(self, split='train', max_length=MAX_SIZE):
        super().__init__()
        self.dataset = load_dataset(
            'wikimedia/wikipedia', '20231101.en', 
            split=split, streaming=False,
            trust_remote_code=True                 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self._prep_item(self.dataset[idx])
    
    def _prep_item(self, item):
            text = item['text']
            # Get byte truncated byte sequence
            bytes = text.encode('utf-8')[:self.max_length]
            bytes_tensor = torch.tensor(list(bytes))
            trunc_text = bytes.decode(encoding='utf-8', errors='replace')

            # Get BERT tokens
            tokens = self.tokenizer(
                trunc_text,
                truncation=True, 
                max_length=len(bytes),
                return_tensors='pt'
            )['input_ids'][0]
            if len(bytes) < tokens.size(0):
                print(text)
                print(f"Bytes: {len(bytes)}, Tokens: {tokens.size(0)}")
                bytes_tensor = F.pad(bytes_tensor, (0, tokens.size(0) - len(bytes)))
            else:
                # Right pad tokens with stop token s.t. it shares length with bytes
                tokens = F.pad(tokens, (0, len(bytes) -tokens.size(0)), value=self.tokenizer.pad_token_id)
            assert bytes_tensor.shape == tokens.shape
            
            #print("Yo! here we are brochenko!", bytes_seq.shape, tokens.shape)
            return {
                'bytes': bytes_tensor,
                'target_tokens': tokens,
                'byte_length': len(bytes),
                'token_length': len(tokens)
            }

    def __iter__(self):
        for item in self.dataset:
            yield self._prep_item(item)

class NeuralTokenizerModule(L.LightningModule):
    def __init__(
        self,
        model_name=MODEL,
        nhead=TXF_HEADS,
        num_layers=TXF_LAYERS,
        max_output_length=MAX_SIZE,
        vocab_size=30522,  # BERT vocab size
        learning_rate=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        mdl = transformers.AutoModel.from_pretrained(model_name)
        self.target_embeddings = mdl.get_input_embeddings()
        # Freeze the bert embeddings
        for param in self.target_embeddings.parameters():
            param.requires_grad = False
        d_model = self.target_embeddings.weight.shape[1]
        self.d_model = d_model
        #print("d_model", d_model)

        # Byte embeddings (0-255)
        self.byte_embeddings = nn.Embedding(256, d_model)
        self.pos_encoder = nn.Embedding(max_output_length, d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
    
    def raw_forward(self, s: str, device="cuda"):
        bytes = s.encode('utf-8')
        bytes_tensor = torch.tensor(list(bytes)).to(device)
        bytes_embs = self.byte_embeddings(bytes_tensor).unsqueeze(0)
        return self.forward(bytes_tensor, bytes_embs)
    
    def forward(self, bytes_seq, bytes_embs, target_tokens=None, target_embs=None):
        # Run through transformer
        output = self.transformer(
            bytes_embs,
            target_embs,
        )
        return output
    
    def _train_val_step(self, batch, batch_idx):
        B, T = batch['bytes'].shape
        bytes_seq = batch['bytes']
        target_tokens = batch['target_tokens']

        # Embed bytes
        positions = torch.arange(bytes_seq.size(1), device=self.device)
        byte_embs = self.byte_embeddings(bytes_seq) + self.pos_encoder(positions)
        # Embed targets
        target_embs = self.target_embeddings(target_tokens)
        assert target_embs.shape == (B, T, self.d_model)
        assert byte_embs.shape == (B, T, self.d_model)
        
        # Run through transformer
        embs = self.forward(bytes_seq, byte_embs, target_tokens, target_embs)
        loss = F.mse_loss(
            embs,
            target_embs
        )
    
        return loss
        

    def training_step(self, batch, batch_idx):
        B, T = batch['bytes'].shape
        loss = self._train_val_step(batch, batch_idx)
        self.log('train_loss', loss, batch_size=B)
        return loss
    
    def validation_step(self, batch, batch_idx):
        B, T = batch['bytes'].shape
        loss = self._train_val_step(batch, batch_idx)
        self.log('val_loss', loss, batch_size=B)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Training setup
def collate_fn(batch):
    # Pad sequences in batch to same length
    #print(len(batch))
    #print(batch[0].keys())
    max_byte_len = max(x['byte_length'] for x in batch)
    max_token_len = max(x['token_length'] for x in batch)
    
    bytes_padded = torch.zeros((len(batch), max_byte_len), dtype=torch.long)
    tokens_padded = torch.zeros((len(batch), max_token_len), dtype=torch.long)
    
    for i, item in enumerate(batch):
        bytes_padded[i, :item['byte_length']] = item['bytes']
        tokens_padded[i, :item['token_length']] = item['target_tokens']
    
    return {
        'bytes': bytes_padded,
        'target_tokens': tokens_padded,
        'byte_length': [x['byte_length'] for x in batch],
        'token_length': [x['token_length'] for x in batch]
    }

def train_tokenizer():
    # Data
    train_dataset = WikipediaByteDataset(split='train[0%:90%]')
    val_dataset = WikipediaByteDataset(split='train[90%:100%]')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=collate_fn,
        num_workers=12,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        collate_fn=collate_fn,
        num_workers=12,
        persistent_workers=True
    )
    
    # Model
    model = NeuralTokenizerModule()
    
    # Training
    trainer = L.Trainer(
        max_epochs=1,
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,
        val_check_interval=1000,
        limit_val_batches=0.001,
        log_every_n_steps=100,
        limit_test_batches=0.01,
        #overfit_batches=0.001,
        logger=WandbLogger(project='neural-tokenizer')
    )
    
    trainer.fit(model, train_loader, val_loader)
    return model

if __name__ == "__main__":
    train_tokenizer()