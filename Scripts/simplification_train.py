import subprocess
import sys

def install_requirements():
    print("Installing required packages...")
    required_packages = [
        'transformers',
        'torch',
        'evaluate',
        'sacrebleu',
        'sacremoses',
        'sentencepiece',
        'nvidia-ml-py3'
    ]
    
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    print("All required packages installed!")

# Install packages
install_requirements()

import torch
from transformers import BertTokenizer, EncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import numpy as np
import gc
import evaluate

# Clear any existing GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

class SimplificationDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        print(f"Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.src_tokens = data['src_tokens']
        self.tgt_tokens = data['tgt_tokens']
        print(f"Loaded {len(self.src_tokens)} examples")

    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.src_tokens[idx]), torch.tensor(self.tgt_tokens[idx])

def validate_batch(model, batch, tokenizer, device):
    try:
        src_tokens, tgt_tokens = batch
        src_tokens = src_tokens.to(device)
        
        attention_mask = (src_tokens != tokenizer.pad_token_id).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=src_tokens,
                attention_mask=attention_mask,
                max_length=80,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True,
                return_dict_in_generate=False
            )
            
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(tgt_tokens, skip_special_tokens=True)
            srcs = tokenizer.batch_decode(src_tokens, skip_special_tokens=True)
            
            # Create list of references for each source
            refs_list = [[ref] for ref in refs]  # Each reference needs to be in a list
            
            # Calculate SARI score
            metric_sari = evaluate.load("sari")
            sari_score = metric_sari.compute(sources=srcs, predictions=preds, references=refs_list)["sari"]
            
            return sari_score
            
    except Exception as e:
        print(f"\nError in validation: {e}")
        return None

def train_model(args):
    print("\n=== Initializing Training ===")
    print(f"Device: {args['device']}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args['model'])
    special_tokens_dict = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Initialize model
    print("Loading model...")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(args['model'], args['model'])
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size
    model.config.max_length = args['max_length']
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    model = model.to(args['device'])
    
    # Create dataloaders
    print("\nPreparing dataloaders...")
    train_loader = DataLoader(
        SimplificationDataset(os.path.join(args['data_dir'], 'train.pkl'), tokenizer),
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        SimplificationDataset(os.path.join(args['data_dir'], 'valid.pkl'), tokenizer),
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])
    scaler = torch.amp.GradScaler()  # Updated to new syntax
    
    best_sari = 0
    best_loss = float('inf')
    patience = args['patience']
    no_improve = 0
    
    print(f"\nStarting training with:")
    print(f"Batch size: {args['batch_size']}")
    print(f"Learning rate: {args['learning_rate']}")
    print(f"Max length: {args['max_length']}")
    
    for epoch in range(args['epochs']):
        print(f"\nEpoch {epoch + 1}/{args['epochs']}")
        
        # Training phase
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc="Training")
        
        for src_tokens, tgt_tokens in train_pbar:
            try:
                src_tokens = src_tokens.to(args['device'])
                tgt_tokens = tgt_tokens.to(args['device'])
                
                attention_mask = (src_tokens != tokenizer.pad_token_id).to(args['device'])
                
                # Mixed precision training
                with torch.amp.autocast(device_type='cuda'):  # Updated to new syntax
                    outputs = model(
                        input_ids=src_tokens,
                        attention_mask=attention_mask,
                        labels=tgt_tokens
                    )
                    
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"\nError in training batch: {e}")
                continue
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Save if loss improved
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            print(f"New best loss: {best_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'tokenizer': tokenizer
            }, args['save_path'].replace('.pt', '_best_loss.pt'))
        
        # Validation phase
        print("\nRunning validation...")
        model.eval()
        valid_sari = []
        
        for batch in tqdm(valid_loader, desc="Validation"):
            sari_score = validate_batch(model, batch, tokenizer, args['device'])
            if sari_score is not None:
                valid_sari.append(sari_score)
        
        if valid_sari:
            avg_sari = np.mean(valid_sari)
            print(f"Validation SARI: {avg_sari:.4f}")
            
            if avg_sari > best_sari:
                best_sari = avg_sari
                print(f"New best SARI: {best_sari:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sari': best_sari,
                    'tokenizer': tokenizer
                }, args['save_path'])
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break

if __name__ == "__main__":
    args = {
        'model': 'bert-base-uncased',
        'data_dir': '/content/drive/MyDrive/ANLP/Processed',
        'save_path': '/content/drive/MyDrive/ANLP/Model/best_model.pt',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'learning_rate': 2e-5,
        'epochs': 20,
        'max_length': 80,
        'patience': 4
    }
    
    train_model(args)