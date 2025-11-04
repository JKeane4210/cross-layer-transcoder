from openclt.open_cross_layer_transcoder import OpenCrossLayerTranscoder, ReplacementModel
from sklearn.model_selection import train_test_split
from torchinfo import summary
import torch

def create_addition_datset(A: list[int], B: list[int]):
    dataset = set()
    for a in A:
        for b in B:
            dataset.add(f"{a} + {b} =")
    return list(dataset)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the cross-layer transcoder
transcoder = OpenCrossLayerTranscoder(
    model_name="gpt2",  # GPT-2 Small
    num_features=768,   # Number of interpretable features # originally 100
    device=device
)

base_model = transcoder.base_model
# Print base model layer summary

# Train the transcoder on sample texts
# train_texts = [
#     "The capital of France is Paris, which is known for the Eiffel Tower.",
#     "New York City is the largest city in the United States.",
#     # Add more training texts...
# ]

train_texts, test_texts = train_test_split(
    create_addition_datset(
        list(range(1, 25)), 
        list(range(1, 25))
    ),
    test_size=0.2,
    random_state=42
)


metrics = transcoder.train_transcoder(
    texts=train_texts,
    batch_size=2,
    num_epochs=3,
    learning_rate=1e-4
)

# Visualize feature activations for a test text
test_text = test_texts[0]
print(f"Test text: {test_text}")
transcoder.visualize_feature_activations(
    text=test_text,
    top_n=5,
    save_path='feature_activations.png'
)

# Create an attribution graph
transcoder.create_attribution_graph(
    text=test_text,
    threshold=0.1,
    save_path='attribution_graph.png'
)

# Create a replacement model
replacement_model = ReplacementModel(
    base_model_name="gpt2",
    transcoder=transcoder
)

# Generate text with the replacement model
generated_text = replacement_model.generate(
    text=test_text,
    max_length=50
)
print(generated_text)

generated_tokens_base_model = replacement_model.base_model.generate(
    input_ids=transcoder.tokenizer.encode(test_text, return_tensors='pt').to(device),
    max_length=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    attention_mask=None,
    pad_token_id=transcoder.tokenizer.eos_token_id
)
generated_text_base_model = transcoder.tokenizer.decode(generated_tokens_base_model[0], skip_special_tokens=True)
print(generated_text_base_model)

# Save the trained transcoder
transcoder.save_model('cross_layer_transcoder_gpt2.pt')
summary(base_model, input_size=(1, 10), dtypes=[torch.long])
summary(transcoder, input_size=(1, 10), dtypes=[torch.long])
