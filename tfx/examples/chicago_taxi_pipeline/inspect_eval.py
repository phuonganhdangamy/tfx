import tensorflow_model_analysis as tfma

# Load the eval result
eval_result = tfma.load_eval_result('pipelines/chicago_taxi_native_keras/Evaluator/evaluation/8')

# Print the metrics
for (slice_key, metrics) in eval_result.slicing_metrics:
    print(f"Slice: {slice_key}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
