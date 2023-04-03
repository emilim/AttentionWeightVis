from neo4j import GraphDatabase
import random
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text
import sys

model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)


def print_translation(sentence, tokens):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')

def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)

translator = tf.saved_model.load('translator')
sentence = sys.argv[1]  

translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
print_translation(sentence, translated_text)

head = 0
# Shape: `(batch=1, num_heads, seq_len_q, seq_len_k)`.
attention_heads = tf.squeeze(attention_weights, 0)
attention = attention_heads[head]

in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.pt.lookup(in_tokens)[0]

plot_attention_head(in_tokens, translated_tokens, attention)
plt.show()

translated_tokens = translated_tokens[1:]


# old method
'''
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path_to_file = tf.keras.utils.get_file("shakespeare.txt", url)

text = open(path_to_file, "rb").read().decode(encoding="utf-8")

chars = sorted(list(set(text))) # unique chars
vocab_size = len(chars)
'''
class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def create_markov_chain_from_attention(self):
        with self.driver.session(database="neo4j") as session:
            result = session.execute_write(self._create_markov_chain_from_attention)
            print("Create Markov Chain from attention")

    @staticmethod
    def _create_markov_chain_from_attention(tx):
        # delete all nodes and relationships
        tx.run("MATCH (n) DETACH DELETE n")
        
        node_query = "CREATE (:State {id: $id})"
        for char in in_tokens:
            tx.run(node_query, id=char.numpy().decode("utf-8"))
        for char in translated_tokens:
            tx.run(node_query, id=char.numpy().decode("utf-8"))
        
        rel_query = """
        MATCH (a:State {id: $a_id})
        MATCH (b:State {id: $b_id})
        CREATE (a)-[:Transition {prob: $prob}]->(b)
        """
        print(attention)
        print(in_tokens)
        print(translated_tokens)
        for i in range(len(in_tokens)):
            for j in range(len(translated_tokens)):
                print(attention[j][i].numpy(), j, i)
                tx.run(rel_query, a_id=in_tokens[i].numpy().decode("utf-8"), b_id=translated_tokens[j].numpy().decode("utf-8"), prob=attention[j][i].numpy())

    # old method
    '''
    @staticmethod
    def _create_markov_chain_from_voc(tx):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        path_to_file = tf.keras.utils.get_file("shakespeare.txt", url)

        text = open(path_to_file, "rb").read().decode(encoding="utf-8")
        text = text[:1000]

        chars = sorted(list(set(text))) # unique chars
        vocab_size = len(chars)

        # Create the nodes in the database
        node_query = "CREATE (:State {id: $id})"
        for char in chars:
            tx.run(node_query, id=char)

        transition_probs = np.zeros([vocab_size, vocab_size])

        for i in range(len(text) - 1):
            char = text[i]
            next_char = text[i + 1]
            char_index = chars.index(char)
            next_char_index = chars.index(next_char)
            transition_probs[char_index, next_char_index] += 1

        # Create the relationships between the nodes
        rel_query = """
        MATCH (a:State {id: $a_id})
        MATCH (b:State {id: $b_id})
        CREATE (a)-[:Transition {prob: $prob}]->(b)
        """
        for i in range(vocab_size):
            for j in range(vocab_size):
                tx.run(rel_query, a_id=chars[i], b_id=chars[j], prob=transition_probs[i][j])

    @staticmethod
    def _create_markov_chain(tx):
        # Define the number of states in the Markov chain
        num_states = 20
        
        # Create the nodes in the database
        node_query = "CREATE (:State {id: $id})"
        for i in range(num_states):
            tx.run(node_query, id=i)

        # Define the transition probabilities between states randomly for num_states
        transition_probs = [[random.random() for _ in range(num_states)] for _ in range(num_states)]

        # Create the relationships between the nodes
        rel_query = """
        MATCH (a:State {id: $a_id})
        MATCH (b:State {id: $b_id})
        CREATE (a)-[:Transition {prob: $prob}]->(b)
        """
        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    prob = transition_probs[i][j]
                    if random.random() <= prob:
                        tx.run(rel_query, a_id=i, b_id=j, prob=prob)

    def markov_blanket(self):
        with self.driver.session(database="neo4j") as session:
            result = session.execute_write(self._markov_blanket)
            print(result)
            return result

    @staticmethod
    def _markov_blanket(tx):
        # Define the structure of the Bayesian network
        model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('B', 'D'), ('B', 'E')])

        # Generate some random data to fit the model
        data = pd.DataFrame({
            'A': np.random.randint(0, 2, 100),
            'B': np.random.randint(0, 2, 100),
            'C': np.random.randint(0, 2, 100),
            'D': np.random.randint(0, 2, 100),
            'E': np.random.randint(0, 2, 100)
        })

        # Fit the model using the Maximum Likelihood Estimator
        model.fit(data, estimator=MaximumLikelihoodEstimator)

        # Get the Markov blanket of variable D
        mb_d = model.get_markov_blanket('D')

        return mb_d
    '''
    

if __name__ == "__main__":
    # Aura queries use an encrypted connection using the "neo4j+s" URI scheme
    uri = "neo4j+s://8837656c.databases.neo4j.io"
    user = "neo4j"
    password = "yluJ3qA-MFAEIIh3O3yVN7iHT3QzaZSiqyIiNK3jByQ"
    app = App(uri, user, password)
    # app.create_friendship("Alice python", "David python")
    # app.find_person("Alice python")
    # app.create_markov_chain()
    app.create_markov_chain_from_attention()
    app.close()