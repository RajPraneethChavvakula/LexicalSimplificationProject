# Neural Text Simplification System

This repository contains the implementation of a text simplification system that uses BERT-based models to make complex text more accessible while preserving its meaning. The system integrates lexical substitution, sentence splitting, and structural reordering techniques.

## System Architecture

The system consists of several key components:

1. **Data Layer**
   - Uses WikiLarge Dataset (296,402 pairs) for training
   - TurkCorpus for validation (2,000 pairs) and testing (359 pairs)

2. **Preprocessing Layer**
   - Text preprocessing and cleaning
   - BERT tokenization
   - Data loading and batching

3. **Training Layer**
   - BERT encoder-decoder architecture
   - AdamW optimizer
   - Mixed precision training

4. **Evaluation Layer**
   - SARI score metric (primary)
   - BLEU score metric (secondary)
   - Best model selection based on both metrics

5. **Post-Processing Layer**
   - Entity restoration
   - Grammar fixing
   - Punctuation normalization
   - Key information preservation

## Results

Our system achieves the following performance metrics:
- SARI Score: 47.53
- BLEU Score: 51.05
- Average Compression Ratio: 0.91

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/text-simplification.git
cd text-simplification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Training

To train the model:
```bash
python simplification_train.py
```

Key training parameters:
- Batch size: 32
- Learning rate: 2e-5
- Max sequence length: 80
- Early stopping patience: 4

### Evaluation

To evaluate a trained model:
```bash
python post_processed_eval.py
```

## File Structure

```
├── simplification_train.py   # Main training script
├── post_processed_eval.py    # Evaluation script
├── preprocess.py            # Data preprocessing
└── requirements.txt         # Package dependencies
```

## Example Output

Input: "The economic outlook, which has been fluctuating due to various global factors, remains uncertain for the foreseeable future."

Output: "The economic outlook is uncertain because of global factors."

## References

1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
2. SARI: A New Evaluation Metric for Text Simplification
3. The TurkCorpus: A Resource for Text Simplification

## License

This project is under the MIT License. See LICENSE file for details.

## Contact

For questions or feedback, please open an issue in the GitHub repository.
