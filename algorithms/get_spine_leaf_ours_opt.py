import sys
import os
import json

# --- Path Setup ---
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from network_generator.spine_leaf_network import SpineLeafNetwork
from solve_spine_leaf import *

# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Network and Jobs from JSON ---
    json_filename = "spine8_leaf8_servers64_jobs8_seed42.json"
    json_file_path = os.path.join(project_root, "generated_data", json_filename)

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    network_json = data['network_json']
    jobs_json = data['jobs_json']

    my_seed = network_json['metadata']['random_seed']

    # --- Create Network Topology from JSON ---
    spine_leaf_network = SpineLeafNetwork(network_json = network_json)

    # --- Initialize jobs from JSON ---
    # The 'workers' in the original jobs list was a set. JSON loads it as a list.
    # To maintain consistency, we convert it back to a set.
    jobs = jobs_json
    for job in jobs:
        job['workers'] = set(job['workers'])

    # --- Solve Spine Leaf Problem By Ours Method ---
    gurobi_params = {
        'OutputFlag': 0,
        'NonConvex': 2,
        'Seed': my_seed,
        'TimeLimit': 1800,
        'MIPFocus': 1,  # Focus on finding feasible solutions quickly
        'Heuristics': 0.1 # Spend 10% of time on heuristics
    }
    
    start_time = time.time()
    solution = solve_spine_leaf_opt(spine_leaf_network, jobs, gurobi_params=gurobi_params)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
    job_links = {}

    for job_id, job_result in solution['jobs'].items():
        job_links[job_id] = set()
        for link, value in job_result['x'].items():
            if value > 0.5:
                job_links[job_id].add(link)

    output_dir = os.path.join(project_root, "routing_results")
    output_filename = f"{json_filename.split('.')[0]}_ours_opt.json"
    output_filepath = os.path.join(output_dir, output_filename)

    generate_job_links_json(job_links, output_filepath)