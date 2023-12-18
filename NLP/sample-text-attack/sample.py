# text-attack end to end
import textattack
import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

# Attack 20 samples with CSV logging and checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(
    num_examples=5,
    log_to_csv="log.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints",
    disable_stdout=True
)

attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
