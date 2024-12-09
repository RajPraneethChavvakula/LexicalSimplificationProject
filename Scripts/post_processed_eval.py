# All imports
import torch
import numpy as np
import pandas as pd
import pickle
import sacrebleu
import evaluate
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, EncoderDecoderModel
import spacy
import re
from typing import List

# Post-processor class
class SimplificationPostProcessor:
    def __init__(self):
        """Initialize post-processor with required models and resources"""
        self.nlp = spacy.load('en_core_web_sm')
        
    def process_batch(self, sources: List[str], predictions: List[str]) -> List[str]:
        """Process a batch of simplifications"""
        improved_predictions = []
        for src, pred in zip(sources, predictions):
            improved = self.process(src, pred)
            improved_predictions.append(improved)
        return improved_predictions
    
    def process(self, source: str, prediction: str) -> str:
        """Process a single simplification"""
        try:
            # Process both texts
            src_doc = self.nlp(source)
            pred_doc = self.nlp(prediction)
            
            # Apply enhancements
            text = prediction
            text = self._restore_entities(text, src_doc, pred_doc)
            text = self._ensure_key_info(text, src_doc)
            text = self._fix_grammar(text)
            text = self._normalize_punctuation(text)
            
            return text
        except Exception as e:
            print(f"Post-processing failed: {str(e)}")
            return prediction
    
    def _restore_entities(self, text: str, src_doc, pred_doc) -> str:
        """Restore named entities from source if missing in prediction"""
        pred_ents = {ent.text.lower() for ent in pred_doc.ents}
        
        for ent in src_doc.ents:
            # Only add important entity types
            if ent.label_ in {'PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL'} and \
               ent.text.lower() not in pred_ents:
                text = f"{text} {ent.text}"
        
        return text
    
    def _ensure_key_info(self, text: str, src_doc) -> str:
        """Ensure key information is preserved"""
        key_elements = []
        for token in src_doc:
            if token.dep_ in {'nsubj', 'dobj'} and not token.is_stop:
                if token.text.lower() not in text.lower():
                    key_elements.append(token.text)
        
        if key_elements:
            text = f"{text} {' '.join(key_elements)}"
        return text
    
    def _fix_grammar(self, text: str) -> str:
        """Fix common grammatical issues"""
        # Fix capitalization
        text = text[0].upper() + text[1:]
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])(?![\s"])', r'\1 ', text)
        
        # Ensure sentence ends with punctuation
        if not text.rstrip()[-1] in '.!?':
            text = text.rstrip() + '.'
            
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation patterns"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix quotes
        text = re.sub(r'``|\'\'', '"', text)
        
        # Normalize dashes
        text = re.sub(r'--+', '—', text)
        
        return text.strip()

# Dataset class
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

def evaluate_bleu_sari(model_path, test_data_path, device='cuda'):
    """Evaluate model with BLEU and SARI metrics, including post-processing"""
    print("Loading model checkpoint...")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    special_tokens_dict = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Initialize model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
    
    # Set model configuration
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size
    model.config.max_length = 80
    model.config.min_length = 10
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Initialize post-processor
    post_processor = SimplificationPostProcessor()

    # Create test dataloader
    test_dataset = SimplificationDataset(test_data_path, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize metrics
    sari_metric = evaluate.load("sari")
    results = {
        'sources': [],
        'predictions': [],
        'post_processed': [],
        'references': [],
        'original_sari': [],
        'improved_sari': [],
        'original_bleu': [],
        'improved_bleu': []
    }

    print("\nEvaluating test set...")
    for batch_idx, (src_tokens, tgt_tokens) in enumerate(tqdm(test_loader)):
        try:
            src_tokens = src_tokens.to(device)
            attention_mask = (src_tokens != tokenizer.pad_token_id).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=src_tokens,
                    attention_mask=attention_mask,
                    decoder_start_token_id=tokenizer.cls_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.sep_token_id,
                    max_length=80,
                    min_length=10,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True
                )

            # Decode outputs
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(tgt_tokens, skip_special_tokens=True)
            srcs = tokenizer.batch_decode(src_tokens, skip_special_tokens=True)

            # Apply post-processing
            improved_preds = post_processor.process_batch(srcs, preds)

            # Calculate scores for each example
            for src, pred, improved_pred, ref in zip(srcs, preds, improved_preds, refs):
                # Store texts
                results['sources'].append(src)
                results['predictions'].append(pred)
                results['post_processed'].append(improved_pred)
                results['references'].append(ref)

                # Calculate original SARI and BLEU
                orig_sari = sari_metric.compute(sources=[src], predictions=[pred], references=[[ref]])["sari"]
                orig_bleu = sacrebleu.corpus_bleu([pred], [[ref]]).score
                results['original_sari'].append(orig_sari)
                results['original_bleu'].append(orig_bleu)

                # Calculate improved SARI and BLEU
                imp_sari = sari_metric.compute(sources=[src], predictions=[improved_pred], references=[[ref]])["sari"]
                imp_bleu = sacrebleu.corpus_bleu([improved_pred], [[ref]]).score
                results['improved_sari'].append(imp_sari)
                results['improved_bleu'].append(imp_bleu)

        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {str(e)}")
            continue

    # Print metrics
    print("\nFinal Metrics:")
    print("Original Model:")
    print(f"  Average SARI: {np.mean(results['original_sari']):.2f}")
    print(f"  Average BLEU: {np.mean(results['original_bleu']):.2f}")
    print("After Post-processing:")
    print(f"  Average SARI: {np.mean(results['improved_sari']):.2f}")
    print(f"  Average BLEU: {np.mean(results['improved_bleu']):.2f}")

    return results

if __name__ == "__main__":
    model_path = '/content/drive/MyDrive/ANLP/Model/best_model.pt'
    test_data_path = '/content/drive/MyDrive/ANLP/Processed/test.pkl'
    
    results = evaluate_bleu_sari(model_path, test_data_path)
    
    # Save results to CSV
    df = pd.DataFrame({
        'source': results['sources'],
        'original_prediction': results['predictions'],
        'post_processed': results['post_processed'],
        'reference': results['references'],
        'original_sari': results['original_sari'],
        'improved_sari': results['improved_sari'],
        'original_bleu': results['original_bleu'],
        'improved_bleu': results['improved_bleu']
    })
    
    df.to_csv('simplification_results_with_postprocessing.csv', index=False)
    print("\nResults saved to simplification_results_with_postprocessing.csv")