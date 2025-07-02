import sys
import os
import json
import time

# --- Path Setup ---
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_script_dir) # This is your 2025_TON_HARouting directory

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from network_generator.fat_tree_network import FatTreeNetwork
from solve_fat_tree import *

# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Network and Jobs from JSON ---
    json_filename = "fat_tree_pod4_servers64_jobs8_seed42.json"
    json_file_path = os.path.join(project_root, "generated_data", json_filename)

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    network_json = data['network_json']
    jobs_json = data['jobs_json']

    my_seed = network_json['metadata']['random_seed']

    # --- Create Network Topology from JSON ---
    fat_tree_network = FatTreeNetwork(network_json=network_json)

    # --- Initialize jobs from JSON ---
    # The 'workers' in the original jobs list was a set. JSON loads it as a list.
    # To maintain consistency, we convert it back to a set.
    jobs = jobs_json
    for job in jobs:
        job['workers'] = set(job['workers'])

    # --- Solve Fat Tree Problem By Ours Method ---
    gurobi_params = {
        'OutputFlag': 0,
        'NonConvex': 2,
        'Seed': my_seed,
        'TimeLimit': 1800, # Increased time limit for potentially harder problem
        'MIPFocus': 1,
        'Heuristics': 0.1,
        # --- Precision-related parameters for performance tuning ---
        #'FeasibilityTol': 1e-3, # Default 1e-6
        #'OptimalityTol': 1e-3,  # Default 1e-6
        #'BarQCPConvTol': 1e-3   # Default 1e-6
    }
    start_time = time.time()
    solution = solve_fat_tree(fat_tree_network, jobs, gurobi_params=gurobi_params)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # --- Save Result to Json File ---
    if solution.get('model_status') != 2: # 2 is GRB.OPTIMAL
        print(f"Warning: Model did not solve to optimality. Status: {solution.get('model_status')}")


    # LP Solution and Integer Solution file to path: ours_results/
    output_dir = os.path.join(project_root, "ours_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # LP solution
    lp_solution_filename = f"{json_filename.split('.')[0]}_ours_lp.json"
    lp_solution_filepath = os.path.join(output_dir, lp_solution_filename)
    lp_solution = get_lp_solution(solution, verbose=False, output_json=True, output_filepath=lp_solution_filepath)

    # Integer solution
    int_solution_filename = f"{json_filename.split('.')[0]}_ours_int.json"
    int_solution_filepath = os.path.join(output_dir, int_solution_filename)
    job_selected_links, jobs_routing_result = score_search(fat_tree_network, jobs, lp_solution)
    get_int_solution_json_file(jobs_routing_result, output_json=True, output_filepath=int_solution_filepath)
    
    # Convert job_selected_links to json file and save to routing_result/ folder
    routing_result_dir = os.path.join(project_root, "routing_results")
    if not os.path.exists(routing_result_dir):
        os.makedirs(routing_result_dir)
        
    routing_result_filename = f"{json_filename.split('.')[0]}_ours.json"
    routing_result_filepath = os.path.join(routing_result_dir, routing_result_filename)
    generate_job_links_json(job_selected_links, routing_result_filepath)

    # --- Print Result ---
    print(f"\n--- Final Result ---")
    print(f"LP Solution saved to {lp_solution_filepath}")
    print(f"Integer Solution saved to {int_solution_filepath}")
    print(f"Routing format solution saved to {routing_result_filepath}")