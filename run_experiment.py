# run_experiment.py

import os
from interpretability.framework import MechanisticInterpretabilityFramework
from interpretability.data import IOIDatasetProcessor
from interpretability.probing import SparseProbeFramework
from interpretability.config import MODEL_NAME, PROBE_CONFIG, TARGET_LAYERS

def main():
    """
    Main function to run the complete mechanistic interpretability experiment.
    """
    print("=== Mechanistic Interpretability Experiment ===")

    # 1. Initialize the model interaction framework
    # Ensure HF_TOKEN is set as an environment variable
    framework = MechanisticInterpretabilityFramework(
        model_name=MODEL_NAME,
        device="auto"
    )

    # 2. Initialize the dataset processor
    dataset_processor = IOIDatasetProcessor()

    # 3. Get data
    print("Loading and preparing data...")
    texts, labels = dataset_processor.get_texts_and_labels(split="train", n_examples=2000)

    # 4. Extract activations from the model
    print(f"Extracting activations for layers: {TARGET_LAYERS}...")
    framework.register_hooks(TARGET_LAYERS, components=["output"])
    activations_dict, attention_mask = framework.extract_activations(
        texts, batch_size=8, max_length=64
    )
    framework.clear_hooks()

    # 5. Initialize and run the sparse probing framework
    print("Starting sparse probing experiment...")
    sparse_framework = SparseProbeFramework(PROBE_CONFIG)
    results = sparse_framework.run_sparse_probing_experiment(
        activations_dict, labels, attention_mask, task_name="IOI_Classification"
    )

    # 6. Visualize and save results
    print("Visualizing and saving results...")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Save raw results
    sparse_framework.save_results(results, os.path.join(output_dir, "probing_results.json"))

    # Generate and save plots
    sparse_framework.visualize_results(results, save_path=os.path.join(output_dir, "summary_plots.png"))
    sparse_framework.visualize_advanced_analysis(results, save_path=os.path.join(output_dir, "advanced_plots.png"))

    # 7. Print summary
    print("\n=== Experiment Summary ===")
    print(f"Task: {results['task_name']}")
    print(f"Layers tested: {len(results['layer_results'])}")
    print(f"Sparsity levels tested: {PROBE_CONFIG.k_values}")

    print("\nBest performing layers:")
    for layer, data in results['summary']['best_layers'].items():
        print(f"  {layer}: Accuracy={data['accuracy']:.3f} (at k={data['best_k']})")

    print("\nProbe type comparison (Mean Accuracy):")
    for probe_type, stats in results['summary']['probe_type_comparison'].items():
        print(f"  {probe_type}: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")

    framework.clear_activations()
    print("\n=== Experiment Complete ===")


if __name__ == "__main__":
    main()
    
    
