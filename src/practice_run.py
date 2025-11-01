from openclt.open_cross_layer_transcoder import OpenCrossLayerTranscoder, ReplacementModel
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the cross-layer transcoder
transcoder = OpenCrossLayerTranscoder(
    model_name="gpt2",  # GPT-2 Small
    num_features=768,   # Number of interpretable features # originally 100
    device=device
)

# Train the transcoder on sample texts
train_texts = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "New York City is the largest city in the United States.",
    # Add more training texts...
]

metrics = transcoder.train_transcoder(
    texts=train_texts,
    batch_size=2,
    num_epochs=3,
    learning_rate=1e-4
)

# Visualize feature activations for a test text
test_text = "The president of the United States lives in the White House."
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
    text="Artificial intelligence",
    max_length=50
)
print(generated_text)

# Save the trained transcoder
transcoder.save_model('cross_layer_transcoder_gpt2.pt')