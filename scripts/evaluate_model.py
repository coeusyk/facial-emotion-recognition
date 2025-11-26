"""Entry point for model evaluation."""
from src.evaluation.evaluate import evaluate_model

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/best_model.pth'
    data_path = sys.argv[2] if len(sys.argv) > 2 else 'data/raw/test'

    evaluate_model(model_path, data_path)