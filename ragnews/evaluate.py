'''
This file is for evaluating the quality of our RAG system using the Hairy Trumpet tool/dataset.
'''
import json
import ragnews
import logging
import argparse

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)

class RAGClassifier:
    def __init__(self):
        """
        Initializes the classifier with valid labels for prediction.
        :param valid_labels: list of valid labels the model can predict
        """
        # self.valid_labels = valid_labels
        pass
    
    def predict(self, masked_text):
        """
        Predict labels for input masked text using the ragnews.rag function.
        :param masked_text: The input sentence containing masked tokens (e.g., "[MASK0] is the democratic nominee")
        :return: The predicted label(s)
        """
        '''
        >>> model = RAGEvaluator()
        >>> model.predict('There no mask token here.')
        []
        >>> model.predict('[MASK0] is the democratic nominee')
        'Harris'
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        '''
        db = ragnews.ArticleDB('ragnews.db')
        # Formulate the prompt for the ragnews.rag function
        textprompt = f'''
        This is a fancier question that is based on standard cloze style benchmarks. I am going to provide you a sentence, and that sentence will have a masked token inside of it that will look like [MASK0] or [MASK1] or [MASK2] and so on.
        And your job is to tell me what the value of that masked token was.

        The size of your output should just be a single word for each mask.
        You should not provide any explanation or other extraneous words. 
        If there are multiple mask tokens, provide each token separately with a whitespace in between. 

        INPUT: [MASK0] is the democratic nominee
        OUTPUT: Harris

        INPUT: [MASK0] is the democratic nominee and [MASK1] is the republican nominee
        OUTPUT: Harris Trump

        INPUT: {masked_text}
        OUTPUT: 
        '''
        
        # Use the ragnews.rag function to predict the labels
        try:
            output = ragnews.rag(textprompt, db, keywords_text=masked_text)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return []

        # Return the predicted labels split by whitespace
        return output.split() if output else []


def evaluate_classifier(data_file_path, classifier):
    """
    Evaluates the classifier on the HairyTrumpet dataset.
    :param data_file_path: Path to the HairyTrumpet dataset.
    :param classifier: An instance of RAGClassifier.
    :return: Accuracy of the classifier.
    """
    correct = 0
    total = 0

    # Open the dataset file
    with open(data_file_path, 'r') as f:
        for line in f:
            dp = json.loads(line)
            masked_text = dp['masked_text']
            true_labels = dp.get('masks', [])
            
            # Run prediction
            predicted = classifier.predict(masked_text)
            
            # Count as correct if the predicted label(s) match any of the true labels
            if isinstance(predicted, str):  # If single prediction
                predicted = [predicted]
            
           # Check if any of the predicted values match the true labels
            if any(p.lower() in [tl.lower() for tl in true_labels] for p in predicted):
                correct += 1

            total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    # Argument parsing for the data file path
    parser = argparse.ArgumentParser(description='Evaluate the RAGClassifier on a HairyTrumpet dataset.')
    parser.add_argument('datafile', type=str, help='Path to the HairyTrumpet data file')

    args = parser.parse_args()

    # Step 1: Initialize the RAGClassifier
    classifier = RAGClassifier()

    # Step 2: Evaluate the classifier on the dataset and compute accuracy
    evaluate_classifier(args.datafile, classifier)

if __name__ == "__main__":
    main()
