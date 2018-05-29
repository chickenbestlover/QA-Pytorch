from allennlp.modules.elmo import Elmo, batch_to_ids
import os

CACHE_ROOT = os.getenv('ALLENNLP_CACHE_ROOT', os.path.expanduser(os.path.join('~', '.allennlp')))
DATASET_CACHE = os.path.join(CACHE_ROOT, "datasets")
print(DATASET_CACHE)
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)
