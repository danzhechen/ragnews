'''
For evaluating the quality of the RAG pipeline using the
Hairy Trumpet dataset.
'''

import json
import logging
import ragnews

import re
from tabulate import tabulate

# Set up logging
logging.basicConfig(level=logging.INFO)

class RAGClassifier:
    def __init__(self, valid_labels):
        '''
        Initialize the RAGClassifier. The __init__ function should take
        an input that specifies the valid labels to predict.
        The class should be a "predictor" following the scikit-learn interface.
        You will have to use the ragnews.rag function internally to perform the prediction.
        '''
        self.valid_labels = valid_labels

    def extract_search_keywords(text, seed=None, temperature=None):
        '''
        This helper function performs keyword extraction for a Cloze-style test.
        It generates important keywords based on the provided masked text to assist in finding relevant articles.
        '''
        system = """You are helping a team do Cloze test. Your task is to write a search query that will retrieve relevant articles from a news database. Craft your query with only the most important keywords related to the masked text. Limit to 5 keywords at most. Return only search keywords, do not include any other text or numbers."""
        # Log the inputs for debugging
        logging.info(f"Calling LLM with system prompt: {system}")
        logging.info(f"Masked text: {text}")
        # Call the LLM to generate keywords
        try:
            keywords = ragnews.run_llm(system, text, seed=seed, temperature=temperature)
            logging.info(f"LLM generated keywords: {keywords}")
            return keywords
        except Exception as e:
            logging.error(f"Error extracting keywords: {e}")
            return "Error: Unable to extract keywords"

    def predict(self, masked_text: str, attempt=0, max_attempts=10):
        '''
        Predict the labels of the documents.

        >>> model = RAGClassifier(['Trump', 'Biden', 'Harris'])
        >>> model.predict('There is no mask token')
        []
        >>> model.predict('[MASK0] is the democratic nominee for president in 2024')
        ['Harris']
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        >>> text1 = """On July 13, 2024, during a campaign rally in Butler, Pennsylvania, presidential candidate [MASK0] was shot at in a failed assassination attempt. The gunfire caused minor damage to [MASK0]'s upper right ear, while one spectator was killed and two others were critically injured. On September 15, 2024, the security detail of [MASK0] spotted an armed man while the former president was touring his golf course in West Palm Beach. They opened fire on the suspect, which fled on a vehicle and was later captured thanks to the contribution of an eyewitness. In the location where the suspect was spotted, the police retrieved an AK-47-style rifle with a scope, two rucksacks and a GoPro."""
        >>> model.predict(text1)
        ['Trump']
        >>> text2 = """After a survey by the Associated Press of Democratic delegates on July 22, 2024, [MASK0] became the new presumptive candidate for the Democratic party, a day after declaring her candidacy. She would become the official nominee on August 5 following a virtual roll call of delegates."""
        >>> model.predict(text2)
        ['Harris']
        '''
        db = ragnews.ArticleDB('ragnews.db')
        
        # Check if there are any mask tokens in the text
        if not re.search(r'\[MASK\d+\]', masked_text):
            return []

        if attempt >= max_attempts:
            logging.error(f"Exceeded maximum retry attempts ({max_attempts}). No valid articles found.")
            return []

        mask_tokens = list(set(re.findall(r'\[MASK\d+\]', masked_text)))
        
        # Dynamically generate example names based on the number of masks
        if len(mask_tokens) > 5:
            dynamic_names = [f'Person{i+1}' for i in range(len(mask_tokens))]
        else:
            dynamic_names = ['Amy', 'Brian', 'Cleo', 'David', 'Eli'][:len(mask_tokens)]

        mapped_names = ', '.join([f'[MASK{i}] is {dynamic_names[i]}' for i in range(len(mask_tokens))])
        result_names = '\n'.join(dynamic_names[:len(mask_tokens)])

        # Choose which system to use based on the task complexity
        system = """
        You are a helpful assistant that predicts the answers of the masked text based only on the context provided. Masked text is in the format of {masks}. The answers to choose from are: {valid_labels}. Think through your answer step-by-step in no more than 50 words. If your answer is a person, provide their last name ONLY. As soon as you have a final answer for all masks, provide each answer on a new line at the end of your response inside a single <answer> tag like the example. Each unique mask id gets a unique answer. DO NOT REPEAT THE ANSWER FOR THE SAME MASK ID.
Example:
...(your reasoning here)...
Therefore {example_mapping}.

<answer>
{example_answers}
</answer>"""
        system = system.format(
            masks=' '.join(mask_tokens),
            example_mapping=mapped_names,
            example_answers=result_names,
            valid_labels=self.valid_labels,
        )

        # Extract keywords using the refactored method
        query_keywords = RAGClassifier.extract_search_keywords(masked_text)
        logging.info(f'Extracted keywords: {query_keywords}')
        
        # Perform the RAG query using the generated keywords
        try:
            rag_output = ragnews.rag(
                masked_text, db, keywords_text=query_keywords, system=system,temperature=0.5, stop='</answer>', max_articles_length=20000, verbose=True
            )

            if 'No articles found' in rag_output:
                logging.warning('No articles found, trying again... attempt: %d', attempt)
                return self.predict(masked_text, attempt=attempt+1)

            if '<answer>' not in rag_output and attempt < 3:
                logging.warning('Output parsing error, retrying... attempt: %d', attempt)
                return self.predict(masked_text, attempt=attempt+1)
            elif '<answer>' not in rag_output and attempt >= 3:
                return []

            # Parse the output into the final results
            output_lines = rag_output.strip().split('<answer>')
            final_results = [line for line in output_lines[-1].split('\n') if line.strip()]

            return final_results

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return []

def extract_search_keywords(text, seed=None, temperature=None):
    """
    This helper function performs keyword extraction for a Cloze-style test.
    It generates important keywords based on the provided masked text to assist in finding relevant articles.
    """
    system = """
    You are helping a team do a Cloze test. Your task is to write a search query that will retrieve relevant articles from a news database.
    Craft your query with only the most important keywords related to the masked text. Limit to 5 keywords at most.
    Return only search keywords, do not include any other text or numbers.
    """
    
    # Log the inputs for debugging
    logging.info(f"Calling LLM to extract keywords for text: {text}")

    try:
        keywords = ragnews.run_llm(system, text, seed=seed, temperature=temperature)
        
        # Check and log what the LLM returned
        if keywords is None or keywords.strip() == "":
            logging.error(f"LLM did not generate valid keywords for text: {text}")
            return None
        
        logging.info(f"LLM generated keywords: {keywords}")
        return keywords
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return None


if __name__ == '__main__':
    import argparse

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    args = parser.parse_args()

    # Load data from the specified file
    with open(args.data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # Extract unique labels from the dataset
    unique_labels = set()
    with open(args.data_file) as fin:
        for idx, line in enumerate(fin):
            record = json.loads(line)
            unique_labels.update(record['masks'])

    # Instantiate the model
    classifier_model = RAGClassifier(unique_labels)

    success_count = 0
    failure_count = 0

    # Evaluate the model on the dataset
    for idx, entry in enumerate(dataset):
        logging.info('Processing entry %d out of %d', idx, len(dataset))
        predicted_labels = classifier_model.predict(entry['masked_text'])
        actual_labels = entry['masks']

        if len(predicted_labels) == len(actual_labels):
            if all(mask.lower() in pred.lower() for mask, pred in zip(actual_labels, predicted_labels)):
                success_count += 1
            else:
                failure_count += 1
        else:
            failure_count += 1

    # Print the success and failure counts
    print(f'Success count: {success_count}')
    print(f'Failure count: {failure_count}')

    # Print accuracy ratio
    total_attempts = success_count + failure_count
    if total_attempts > 0:
        accuracy = success_count / total_attempts
        print(f'Accuracy: {accuracy * 100:.2f}%')
    else:
        print('No data to evaluate.')
