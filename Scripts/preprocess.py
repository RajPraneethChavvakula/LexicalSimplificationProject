import os
import pickle
from tqdm import tqdm
import re
from transformers import BertTokenizer
import torch

class TextPreprocessor:
    def __init__(self, max_length=80):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        special_tokens_dict = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def clean_text(self, text):
        # Basic cleaning
        text = text.strip().lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Basic punctuation normalization
        text = text.replace('..', '.')
        text = text.replace('...', '.')
        text = text.replace('!!', '!')
        text = text.replace('??', '?')
        return text

    def process_batch(self, src_lines, tgt_lines):
        processed_data = {
            'src': [],
            'tgt': [],
            'src_tokens': [],
            'tgt_tokens': []
        }
        
        for src, tgt in zip(src_lines, tgt_lines):
            try:
                # Clean texts
                src = self.clean_text(src)
                tgt = self.clean_text(tgt)
                
                if not src or not tgt:
                    continue
                    
                # Skip if target is longer than source
                if len(tgt.split()) > len(src.split()) * 1.2:
                    continue
                
                # Tokenize
                src_tokens = self.tokenizer.encode(
                    src,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )[0]
                
                tgt_tokens = self.tokenizer.encode(
                    tgt,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )[0]
                
                processed_data['src'].append(src)
                processed_data['tgt'].append(tgt)
                processed_data['src_tokens'].append(src_tokens.tolist())
                processed_data['tgt_tokens'].append(tgt_tokens.tolist())
                
            except Exception as e:
                continue
        
        return processed_data

    def process_files(self, src_file, tgt_file, batch_size=1000):
        print(f"Processing files:\nSource: {src_file}\nTarget: {tgt_file}")
        
        all_processed_data = {
            'src': [],
            'tgt': [],
            'src_tokens': [],
            'tgt_tokens': []
        }
        
        with open(src_file, 'r', encoding='utf-8') as src_f, \
             open(tgt_file, 'r', encoding='utf-8') as tgt_f:
            
            src_lines = []
            tgt_lines = []
            total_processed = 0
            
            # Process in batches
            for src_line, tgt_line in tqdm(zip(src_f, tgt_f)):
                src_lines.append(src_line)
                tgt_lines.append(tgt_line)
                
                if len(src_lines) >= batch_size:
                    batch_data = self.process_batch(src_lines, tgt_lines)
                    
                    # Extend all data lists
                    for key in all_processed_data:
                        all_processed_data[key].extend(batch_data[key])
                    
                    total_processed += len(batch_data['src'])
                    print(f"Processed {total_processed} pairs")
                    
                    # Clear batch
                    src_lines = []
                    tgt_lines = []
            
            # Process remaining lines
            if src_lines:
                batch_data = self.process_batch(src_lines, tgt_lines)
                for key in all_processed_data:
                    all_processed_data[key].extend(batch_data[key])
                total_processed += len(batch_data['src'])
        
        print(f"Total processed pairs: {total_processed}")
        return all_processed_data

def main():
    # Setup paths
    base_dir = "/content/drive/MyDrive/ANLP"
    data_dir = os.path.join(base_dir, "Data")
    processed_dir = os.path.join(base_dir, "Processed")
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Processed directory: {processed_dir}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_length=80)
    
    # Process each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        src_file = os.path.join(data_dir, f"{split}_src.txt")
        tgt_file = os.path.join(data_dir, f"{split}_tgt.txt")
        
        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            print(f"Skipping {split} - files not found")
            continue
        
        try:
            processed_data = preprocessor.process_files(src_file, tgt_file)
            
            # Save processed data
            output_file = os.path.join(processed_dir, f"{split}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(processed_data, f)
            
            print(f"Processed {len(processed_data['src'])} pairs")
            print(f"Saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            continue

if __name__ == "__main__":
    main()