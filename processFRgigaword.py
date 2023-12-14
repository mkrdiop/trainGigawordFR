"""
This code will:

Loop through the Gigaword files in the specified directory.
Extract sentences and perform basic cleaning by removing punctuation and special characters,
converting to lowercase, and splitting into tokens.
Update a vocabulary counter for all encountered words.
Filter and assign sentences to the training and validation datasets based on the file names.
Build a vocabulary dictionary with the most frequent vocab_size words and save it to a JSON file.
Save the preprocessed training and validation data as pickle files.
Remember to adjust the vocab_size parameter according to your needs and available resources. 
You can also modify the preprocessing steps further based on your specific requirements, 
such as applying stemming or lemmatization for improved tokenization.

"""
import os
import re
from collections import Counter

def preprocess_gigaword_french(data_dir, output_dir):
  """
  Preprocesses the French Gigaword dataset for training a DRGD model.

  Args:
    data_dir: Path to the directory containing the downloaded Gigaword files.
    output_dir: Path to the directory to store the preprocessed data.

  Returns:
    vocab: A dictionary mapping words to their integer indices.
    train_data: A list of lists of integer token IDs for the training sentences.
    val_data: A list of lists of integer token IDs for the validation sentences.

  """

  vocab = Counter()
  train_data = []
  val_data = []

  # Extract sentences from Gigaword files
  for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename), 'r') as f:
      for line in f:
        # Filter out empty lines and sentence boundaries
        if line.strip() and not line.startswith('<'):
          sentence = line.strip().lower()

          # Clean and tokenize the sentence
          sentence = re.sub(r'[^\w\s]', ' ', sentence)
          tokens = sentence.split()

          # Update vocabulary and add sentence to data split
          vocab.update(tokens)
          if filename.startswith('train'):
            train_data.append([vocab[token] for token in tokens])
          elif filename.startswith('val'):
            val_data.append([vocab[token] for token in tokens])

  # Build and save vocabulary
  vocab = {w: i for i, w in enumerate(vocab.most_common(vocab_size))}
  with open(os.path.join(output_dir, 'vocabFR.json'), 'w') as f:
    json.dump(vocab, f)

  # Save preprocessed data
  with open(os.path.join(output_dir, 'train_data_FR.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
  with open(os.path.join(output_dir, 'val_data_FR.pkl'), 'wb') as f:
    pickle.dump(val_data, f)

  return vocab, train_data, val_data

# Define the desired vocabulary size
vocab_size = 50000

# Call the preprocessing function
vocab, train_data, val_data = preprocess_gigaword_french(data_dir, output_dir)

