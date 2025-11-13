from __future__ import print_function
import sys
import keras
import numpy as np

# Optional: Add visualization capability (set to False for faster inference)
VISUALIZE = True
if VISUALIZE:
    import plotly.graph_objects as go

# Parse command line arguments
model_path = sys.argv[1]
data_path = sys.argv[2]

# Load the trained model
print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)

# Load and preprocess the data
print(f"Loading data from: {data_path}")
rxs = np.fromfile(data_path, dtype=np.uint16).astype('float32')

# Normalize the data (same as training)
rxs -= rxs.mean()
rxs /= rxs.std() + 0.0001

# Reshape for model input (batch_size, height, width, channels)
rxs = np.reshape(rxs, (-1, 256, 256, 1), 'C')

print(f"Running inference on {rxs.shape[0]} image(s)...")

# Make predictions
predictions = model.predict(rxs)

# Display results
print("\n" + "=" * 50)
print("INFERENCE RESULTS")
print("=" * 50)
for i, pred in enumerate(predictions):
    confidence = pred[0] * 100
    result = "DEFECT DETECTED" if pred[0] > 0.5 else "NO DEFECT"
    print(f"Image {i + 1}: {result} (Confidence: {confidence:.2f}%)")
print("=" * 50 + "\n")

# Raw predictions
print("Raw prediction values:")
print(predictions)

# Optional: Create interactive visualization
if VISUALIZE and len(predictions) > 1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(predictions) + 1)),
        y=predictions.flatten(),
        mode='markers+lines',
        marker=dict(
            size=10,
            color=predictions.flatten(),
            colorscale='RdYlGn_r',  # Red for defects, green for no defects
            showscale=True,
            colorbar=dict(title="Defect Probability")
        ),
        line=dict(color='lightblue', width=1),
        name='Predictions'
    ))

    #  Add threshold line at 0.5
    # fig.add_hline(y=0.5, line_dash="dash", line_color="red",
    #               annotation_text="Decision Threshold")

    fig.update_layout(
        title='Defect Detection Predictions',
        xaxis_title='Image Number',
        yaxis_title='Defect Probability',
        yaxis_range=[0, 1],
        template='plotly_white',
        height=500
    )

    # Save as HTML
    output_file = 'inference_results.html'
    fig.write_html(output_file)
    print(f"\nVisualization saved to: {output_file}")
    fig.show()
