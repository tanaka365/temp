# text-attackを用いた摂動の作成（日本語）
import textattack
import transformers
from textattack.shared import GensimWordEmbedding
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
import gensim
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR


model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
    model, tokenizer
)

gensim_model = gensim.models.KeyedVectors.load_word2vec_format(
    "model.vec", binary=False
)
word_embedding = GensimWordEmbedding(gensim_model)
transformation = WordSwapEmbedding(
    max_candidates=100, embedding=word_embedding
)
constraints = [
    RepeatModification(),
    StopwordModification(),
    WordEmbeddingDistance(min_cos_sim=0.7),
]
goal_function = UntargetedClassification(model_wrapper)
# search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack
attack = textattack.Attack(
    goal_function, constraints, transformation, search_method
)

input_text = "私は勉強が楽しいと思っている。"
label = 1  # Positive
attack_result = attack.attack(input_text, label)

print(attack_result.original_result)
print(attack_result.perturbed_result)
