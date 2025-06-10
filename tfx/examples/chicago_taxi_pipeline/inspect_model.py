#!/usr/bin/env python3
"""Script to inspect and test your trained TFX model."""

import tensorflow as tf
import numpy as np
import json

# Path to your trained model
MODEL_PATH = "serving_model/chicago_taxi_native_keras/1748531456"

def inspect_model():
    """Load and inspect the trained model."""
    print("=" * 50)
    print("LOADING AND INSPECTING MODEL")
    print("=" * 50)
    
    # Load the saved model
    model = tf.saved_model.load(MODEL_PATH)
    
    # Print available signatures
    print("Available signatures:")
    for signature_key in model.signatures.keys():
        print(f"  - {signature_key}")
    
    # Get the serving signature
    serving_fn = model.signatures['serving_default']
    
    # Print input/output specs
    print("\nServing signature input spec:")
    for input_key, input_spec in serving_fn.structured_input_signature[1].items():
        print(f"  {input_key}: {input_spec}")
    
    print("\nServing signature output spec:")
    for output_key, output_spec in serving_fn.structured_outputs.items():
        print(f"  {output_key}: {output_spec}")
    
    return model

def create_test_example():
    """Create a test example for prediction."""
    print("\n" + "=" * 50)
    print("CREATING TEST EXAMPLE")
    print("=" * 50)
    
    # Create a sample TensorFlow Example
    example = tf.train.Example()
    
    # Add features (these should match your original data schema)
    features = {
        'trip_start_hour': 14,
        'trip_start_day': 3,
        'trip_start_month': 6,
        'trip_start_timestamp': 1446554700,
        'pickup_census_tract': 1001,
        'dropoff_census_tract': 1002,
        'pickup_community_area': 32,
        'dropoff_community_area': 33,
        'trip_miles': 5.2,
        'fare': 12.5,
        'trip_seconds': 900,
        'pickup_latitude': 41.8781,
        'pickup_longitude': -87.6298,
        'dropoff_latitude': 41.8881,
        'dropoff_longitude': -87.6198,
        'payment_type': 'Credit Card',
        'company': 'Yellow Cab',
        'tips': 0
    }
    
    # Convert to TensorFlow Example format
    for key, value in features.items():
        if isinstance(value, (int, float)):
            if isinstance(value, int):
                example.features.feature[key].int64_list.value.append(value)
            else:
                example.features.feature[key].float_list.value.append(value)
        else:
            example.features.feature[key].bytes_list.value.append(value.encode('utf-8'))
    
    # Serialize the example
    serialized_example = example.SerializeToString()
    print(f"Created test example with features: {list(features.keys())}")
    
    return serialized_example

def make_prediction(model, serialized_example):
    """Make a prediction using the model."""
    print("\n" + "=" * 50)
    print("MAKING PREDICTIONS")
    print("=" * 50)
    
    try:
        # Get the serving function
        serving_fn = model.signatures['serving_default']
        
        # Make prediction
        predictions = serving_fn(examples=tf.constant([serialized_example]))
        
        print("Prediction successful!")
        print(f"Model output keys: {list(predictions.keys())}")
        
        # Extract the prediction value
        output_values = predictions['outputs'].numpy()
        print(f"Prediction values: {output_values}")
        print(f"Prediction shape: {output_values.shape}")
        
        # Interpret the result (assuming binary classification for tipping)
        prob = output_values[0][0]  # Get the probability
        prediction = "High Tipper" if prob > 0.5 else "Low Tipper"
        confidence = prob if prob > 0.5 else 1 - prob
        
        print(f"\nResult: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Probability of high tip: {prob:.3f}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("This might be due to feature schema mismatch.")

def explore_pipeline_artifacts():
    """Explore other pipeline artifacts."""
    print("\n" + "=" * 50)
    print("EXPLORING PIPELINE ARTIFACTS")
    print("=" * 50)
    
    import os
    
    # Check for other pipeline outputs
    pipeline_dir = "/users/amydp/tfx/pipelines/chicago_taxi_native_keras"
    
    if os.path.exists(pipeline_dir):
        print("Pipeline artifacts found:")
        for component in os.listdir(pipeline_dir):
            component_path = os.path.join(pipeline_dir, component)
            if os.path.isdir(component_path):
                print(f"  - {component}/")
                # List recent runs
                try:
                    runs = sorted([d for d in os.listdir(component_path) 
                                 if os.path.isdir(os.path.join(component_path, d))])
                    if runs:
                        print(f"    Latest runs: {runs[-3:]}")  # Show last 3 runs
                except:
                    pass
    else:
        print("Pipeline directory not found at expected location.")

def check_model_metrics():
    """Check if there are any evaluation metrics saved."""
    print("\n" + "=" * 50)
    print("CHECKING EVALUATION METRICS")
    print("=" * 50)
    
    # Look for evaluation results
    eval_dir = "/users/amydp/tfx/pipelines/chicago_taxi_native_keras/Evaluator"
    
    import os
    if os.path.exists(eval_dir):
        print("Evaluator artifacts found. Check for metrics in:")
        for item in os.listdir(eval_dir):
            print(f"  - {eval_dir}/{item}")
    else:
        print("No evaluator directory found.")

if __name__ == "__main__":
    print("TFX Model Inspection and Testing")
    print("================================")
    
    try:
        # 1. Load and inspect model
        model = inspect_model()
        
        # 2. Create test example
        test_example = create_test_example()
        
        # 3. Make prediction
        make_prediction(model, test_example)
        
        # 4. Explore pipeline artifacts
        explore_pipeline_artifacts()
        
        # 5. Check metrics
        check_model_metrics()
        
        print("\n" + "=" * 50)
        print("INSPECTION COMPLETE!")
        print("=" * 50)
        print("\nYour model is ready for serving!")
        print("You can now deploy it using TensorFlow Serving or")
        print("integrate it into your applications.")
        
    except Exception as e:
        print(f"Error during inspection: {e}")
        print("\nTip: Make sure you're running this from the correct directory")
        print("and that the model path exists.")