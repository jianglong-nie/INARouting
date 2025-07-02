import sys
import os
import json
import time

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
        'TimeLimit': 1200,
        'MIPFocus': 1,  # Focus on finding feasible solutions quickly
        'Heuristics': 0.1 # Spend 10% of time on heuristics
    }
    start_time = time.time()
    solution = solve_spine_leaf(spine_leaf_network, jobs, gurobi_params=gurobi_params)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
    
    # --- Save Result to Json File ---

    # LP Solution and Integer Solution file to path: ours_relax_routing/
    output_dir = os.path.join(project_root, "ours_results")
    
    # LP solution
    lp_solution_filename = f"{json_filename.split('.')[0]}_ours_lp.json"
    lp_solution_filepath = os.path.join(output_dir, lp_solution_filename)
    lp_solution = get_lp_solution(solution, verbose=False, output_json=True, output_filepath = lp_solution_filepath)

    # Integer solution
    int_solution_filename = f"{json_filename.split('.')[0]}_ours_int.json"
    int_solution_filepath = os.path.join(output_dir, int_solution_filename)
    job_selected_links, jobs_routing_result = score_search(spine_leaf_network, jobs, lp_solution)
    get_int_solution_json_file(jobs_routing_result, output_json=True, output_filepath=int_solution_filepath)
    
    # Convert job_selected_links to json file and save to routing_result/ folder
    routing_result_dir = os.path.join(project_root, "routing_results")
    routing_result_filename = f"{json_filename.split('.')[0]}_ours.json"
    routing_result_filepath = os.path.join(routing_result_dir, routing_result_filename)
    generate_job_links_json(job_selected_links, routing_result_filepath)

    # --- Print Result ---
    print(f"\n--- Final Result ---")
    print(f"LP Solution saved to {lp_solution_filepath}")
    print(f"Integer Solution saved to {int_solution_filepath}")
    print(f"Routing format solution saved to {routing_result_filepath}")