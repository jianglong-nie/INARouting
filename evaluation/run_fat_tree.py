import os
from utils import load_json_file, water_filling_fat_tree

# Main execution block
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    # --- Configuration ---
    # You can switch between different network configurations by uncommenting the desired line.
    network_data_filename = "fat_tree_pod4_servers64_jobs8_seed42.json"

    baseline_names = ['ours', 'ours_opt']
    baseline_total_throughputs = {}

    print(f"--- Running evaluation for: {network_data_filename} ---")

    for baseline_name in baseline_names:
        results_filename = network_data_filename.replace('.json', f'_{baseline_name}.json')

        network_data_filepath = os.path.join(workspace_root, "generated_data", network_data_filename)
        results_filepath = os.path.join(workspace_root, "routing_results", results_filename)

        network_config = load_json_file(network_data_filepath)
        results = load_json_file(results_filepath)

        if not network_config:
            print(f"Error: Could not load network data file: {network_data_filepath}")
            continue
        if not results:
            print(f"Warning: Could not load results file for baseline '{baseline_name}': {results_filepath}")
            baseline_total_throughputs[baseline_name] = 0
            continue

        # Use the water_filling_fat_tree function to calculate throughput
        job_throughputs = water_filling_fat_tree(network_config, results)
        
        if job_throughputs and 0.0 in job_throughputs.values():
            print(f"Error: Job with zero throughput found for baseline '{baseline_name}'")
            total_max_throughput = 0
        else:
            total_max_throughput = sum(job_throughputs.values())

        baseline_total_throughputs[baseline_name] = total_max_throughput
    
    print("\n--- Evaluation Summary ---")
    # Print out the total throughput and standard deviation for each baseline
    for baseline_name in baseline_names:
        total_throughput = baseline_total_throughputs.get(baseline_name, 'N/A')
        if isinstance(total_throughput, (int, float)):
            print(f"Baseline: {baseline_name:<8} | Total Throughput: {total_throughput:<12.2f}")
        else:
            print(f"Baseline: {baseline_name:<8} | Data not available")