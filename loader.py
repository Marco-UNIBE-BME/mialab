import joblib  

def load_pipeline_predictions(path_inference_output:str) -> tuple:
    with open(path_inference_output, 'rb') as f:
            (images_test, images_prediction, images_probabilities) = joblib.load(f)
    
    return (images_test, images_prediction, images_probabilities)