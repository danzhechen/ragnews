'''
This file is for evaluating the quality of our RAG system using the Hairy Trumpet tool/dataset.
'''
import json
import ragnews
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)

class RAGClassifier:
    def __init__(self, valid_labels):
        """
        Initializes the classifier with valid labels for prediction.
        :param valid_labels: list of valid labels the model can predict
        """
        self.valid_labels = valid_labels

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
        OUTPUT: '''
        
        # Use the ragnews.rag function to predict the labels
        output = ragnews.rag(textprompt, db, keywords_text=masked_text)

        return output

class RAGEvaluator:
    def predict(self, masked_text):
        '''
        >>> model = RAGEvaluator()
        >>> model.predict('There no mask token here.')
        []
        >>> model.predict('[MASK0] is the democratic nominee')
        'Harris'
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        '''
        # you might think about:
        # calling the ragnews.run llm function directly;
        # so we will call the ragnews.rag function
        
        db = ragnews.ArticleDB('ragnews.db')
        textprompt = f'''
This is a fancier question that is based on standard cloze style benchmarks. 
I am going to provide you a sentence, and that sentence will have a masked token inside of it that will look like [MASK0] or [MASK1] or [MASK2] and so on.
And your job is to tell me what the value of that masked token was.

The size of your output should just be a single word for each mask.
You should not provide any explanation or other extraneous words. 
If there are multiple mask tokens, provide each token separately with a whitespace in between. 

INPUT: [MASK0] is the democratic nominee
OUTPUT: Harris

INPUT: [MASK0] is the democratic nominee and [MASK1] is the republican nominee
OUTPUT: Harris Trump

INPUT: {masked_text}
OUTPUT: '''
        output = ragnews.rag(textprompt, db, keywords_text=masked_text)
        return output

def extract_labels(data_file_path):
    labels_set = set()
    with open(data_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            labels = data.get('labels', [])
            labels_set.update(labels)
    return list(labels_set)


if __name__ == "__main__":
    main()
