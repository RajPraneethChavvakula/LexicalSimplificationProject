
from src.data.data_loader import load_simplewiki_data
from src.models.simplification_model import TextSimplificationModel
from src.utils.metrics import calculate_sari

def main():
    # Load data
    data = load_simplewiki_data()
    
    # Initialize model
    model = TextSimplificationModel()
    
    # TODO: Implement training and evaluation pipeline
    
if __name__ == "__main__":
    main()
