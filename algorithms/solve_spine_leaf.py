import gurobipy as gp
from gurobipy import GRB
import time
import json
from typing import Optional, Dict, List
from network_generator.spine_leaf_network import SpineLeafNetwork

# --- Multi-Job Solver Function (with IIS Debugging Emphasis) ---
def solve_spine_leaf_opt(network: SpineLeafNetwork,
                    jobs: List[Dict],
                    gurobi_params: Optional[dict] = None):
    """
    Solves the multi-job max sum-throughput problem using Gurobi.
    (Corrected x variable definition)

    Args:
        network: The SpineLeafNetwork instance.
        jobs: A list of job information dictionaries.
        gurobi_params: Optional Gurobi environment parameters.

    Returns:
        dict: solution_details.
    """
    solution_details = {'model_status': None, 'solve_time': None, 
                        'total_max_throughput': 0.0, 'jobs': {}}
    start_time = time.time()

    job_ids = [job['id'] for job in jobs]
    job_map = {job['id']: job for job in jobs} # Quick lookup

    # --- Pre-calculate job-specific links ---
    job_specific_links = {
        job['id']: network.get_job_specific_links(job) for job in jobs
    }
    valid_x_indices = [
        (j, link)
        for j in job_ids
        for link in job_specific_links.get(j, set())
    ]
    all_relevant_links = set().union(*job_specific_links.values())
    sorted(list(all_relevant_links))

    try:
        with gp.Env(params=gurobi_params) as env:
            with gp.Model(env=env, name="SpineLeafMaxSumThroughput") as model:
                # --- Decision Variables ---
                x = model.addVars(valid_x_indices, vtype=GRB.BINARY, name="x")
                y = model.addVars(job_ids, network.switches, vtype=GRB.BINARY, name="y")
                a = model.addVars(job_ids, network.switches, vtype=GRB.BINARY, name="a")

                # Set a small positive lower bound for lambda_j to enforce lambda_j > 0
                lambda_j = model.addVars(job_ids, vtype=GRB.CONTINUOUS, lb=0, name="lambda")

                # --- Objective Function ---
                model.setObjective(gp.quicksum(lambda_j[j] for j in job_ids), GRB.MAXIMIZE)

                # --- Constraints ---
                M = len(network.all_servers)

                # --- Calculate M for Fair Sharing Constraints ---
                max_link_bw = 0
                for bw in network.link_bandwidths.values():
                    if bw != float('inf'):
                        max_link_bw = max(max_link_bw, bw)
                max_switch_cap = 0
                if network.switch_processing_capacities: # Check if dict is not empty
                    max_switch_cap = max(network.switch_processing_capacities.values())
                num_jobs = len(job_ids)
                # A sufficiently large M value
                M_fair_share = (max(max_link_bw, max_switch_cap) * num_jobs) + 1

                for j in job_ids:
                    job = job_map[j]
                    workers = job['workers']
                    ps = job['ps']
                    ps_leaf = job['ps_leaf']
                    current_job_links = job_specific_links[j]

                    # 1. Worker Start Constraint
                    for w in workers:
                        links_from_w = {link for link in current_job_links if link[0] == w}
                        for link in links_from_w:
                            tor_switch = link[1]
                            model.addConstr(x[j, link] == 1)
                            model.addConstr(y[j, tor_switch] == 1)
                            #gp.quicksum(x[j, link_from_tor] for link_from_tor in links_from_tor_switch) == 1

                    # 2. PS End Constraint
                    model.addConstr(x[j, (ps_leaf, ps)] == 1)
                    model.addConstr(y[j, ps_leaf] == 1)

                    # 2. Flow Conservation/Output
                    for node in network.switches:
                        # input_flow -> y[j, node] and a[j, node]
                        links_to_node = {link for link in current_job_links if link[1] == node}
                        if links_to_node:
                            in_flow_sum_j = gp.quicksum(x[j, link] for link in links_to_node)
                            model.addConstr(in_flow_sum_j >= y[j, node])
                            model.addConstr(in_flow_sum_j <= M * y[j, node])

                            # aggregation definition
                            model.addConstr(in_flow_sum_j >= 2 * a[j, node])
                            model.addConstr(in_flow_sum_j <= 1 + M * a[j, node])

                        # output_flow = y[j, node]
                        links_from_node = {link for link in current_job_links if link[0] == node}
                        if links_from_node:
                            out_flow_sum_j = gp.quicksum(x[j, link] for link in links_from_node)
                            model.addConstr(out_flow_sum_j == y[j, node])

                # --- Shared Resource Constraints (FAIR SHARING MODEL) ---

                # 5. Fair Link Bandwidth Constraint
                # If a link is used by n jobs, the throughput of each of these jobs is limited by link_bandwidth / n.
                for j in job_ids:
                    for link in all_relevant_links:
                        # Skip if link has infinite capacity or job j cannot use this link
                        link_bw = network.link_bandwidths.get(link)
                        if link_bw is None or link_bw == float('inf'):
                            continue
                        if (j, link) not in x:
                            continue # x[j, link] does not exist

                        # Number of jobs using this link
                        job_num_link = gp.quicksum(x[k, link] for k in job_ids if (k, link) in x)

                        # If x[j, link] = 1, then lambda_j * job_num_link <= link_bw.
                        # This is linearized using a big-M formulation, active only when x[j, link] is 1.
                        model.addConstr(lambda_j[j] * job_num_link <= link_bw + M_fair_share * (1 - x[j, link]),
                                        name=f"fair_link_bw_{j}_{link[0]}_{link[1]}")

                # 6. Fair Switch Aggregation Capacity Constraint
                # If a switch is used for aggregation by n jobs, the throughput of each is limited by switch_capacity / n.
                for j in job_ids:
                    for node in network.switches:
                        switch_cap = network.switch_processing_capacities.get(node)
                        if switch_cap is None:
                            continue # Should not happen if network is well-defined
                        if (j, node) not in a:
                            continue # a[j, node] does not exist
                        
                        # Number of jobs aggregating at this switch
                        job_num_agg = gp.quicksum(a[k, node] for k in job_ids if (k, node) in a)

                        # If a[j, node] = 1, then lambda_j * job_num_agg <= switch_cap.
                        # Linearized using big-M, active only when a[j, node] is 1.
                        model.addConstr(lambda_j[j] * job_num_agg <= switch_cap + M_fair_share * (1 - a[j, node]),
                                        name=f"fair_switch_cap_{j}_{node}")

                # --- Solve ---
                model.optimize()

                # --- Extract Results ---
                solve_time = time.time() - start_time
                solution_details['solve_time'] = solve_time
                solution_details['model_status'] = model.Status

                if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
                    solution_details['total_max_throughput'] = model.ObjVal
                    for j in job_ids:
                        job_result = {}
                        job_result['id'] = j
                        job_result['throughput'] = lambda_j[j].X
                        job_specific_link_set = job_specific_links.get(j, set())
                        job_result['x'] = {link: x[j, link].X for link in job_specific_link_set if (j, link) in x}
                        job_result['y'] = {node: y[j, node].X for node in network.switches if (j,node) in y} # Check index exists
                        job_result['a'] = {node: a[j, node].X for node in network.switches if (j,node) in a} # Check index exists
                        solution_details['jobs'][j] = job_result
                elif model.Status == GRB.INFEASIBLE:
                    print("Model is infeasible. No solution exists.")
                else:
                    print(f"Optimization finished with status: {model.Status}")
    except gp.GurobiError as e:
        print(f'A Gurobi error occurred: {e}')
        solution_details['model_status'] = f'GurobiError: {e}'
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        solution_details['model_status'] = f'Exception: {e}'

    return solution_details

# --- Multi-Job Solver Function (with IIS Debugging Emphasis) ---
def solve_spine_leaf(network: SpineLeafNetwork,
                    jobs: List[Dict],
                    gurobi_params: Optional[dict] = None):
    """
    Solves the multi-job max sum-throughput problem using Gurobi.
    (Corrected x variable definition)

    Args:
        network: The SpineLeafNetwork instance.
        jobs: A list of job information dictionaries.
        gurobi_params: Optional Gurobi environment parameters.

    Returns:
        dict: solution_details.
    """
    solution_details = {'model_status': None, 'solve_time': None, 
                        'total_max_throughput': 0.0, 'jobs': {}}
    start_time = time.time()

    job_ids = [job['id'] for job in jobs]
    job_map = {job['id']: job for job in jobs} # Quick lookup

    # --- Pre-calculate job-specific links ---
    job_specific_links = {
        job['id']: network.get_job_specific_links(job) for job in jobs
    }
    valid_x_indices = [
        (j, link)
        for j in job_ids
        for link in job_specific_links.get(j, set())
    ]
    all_relevant_links = set().union(*job_specific_links.values())
    sorted(list(all_relevant_links))

    try:
        with gp.Env(params=gurobi_params) as env:
            with gp.Model(env=env, name="SpineLeafMaxSumThroughput") as model:
                # --- Decision Variables ---
                x = model.addVars(valid_x_indices, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
                y = model.addVars(job_ids, network.switches, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
                a = model.addVars(job_ids, network.switches, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="a")

                # Set a small positive lower bound for lambda_j to enforce lambda_j > 0
                lambda_j = model.addVars(job_ids, vtype=GRB.CONTINUOUS, lb=0, name="lambda")

                # --- Objective Function ---
                model.setObjective(gp.quicksum(lambda_j[j] for j in job_ids), GRB.MAXIMIZE)

                # --- Constraints ---
                M = len(network.all_servers)

                # --- Calculate M for Fair Sharing Constraints ---
                max_link_bw = 0
                for bw in network.link_bandwidths.values():
                    if bw != float('inf'):
                        max_link_bw = max(max_link_bw, bw)
                max_switch_cap = 0
                if network.switch_processing_capacities: # Check if dict is not empty
                    max_switch_cap = max(network.switch_processing_capacities.values())
                num_jobs = len(job_ids)
                # A sufficiently large M value
                M_fair_share = (max(max_link_bw, max_switch_cap) * num_jobs) + 1

                for j in job_ids:
                    job = job_map[j]
                    workers = job['workers']
                    ps = job['ps']
                    ps_leaf = job['ps_leaf']
                    current_job_links = job_specific_links[j]

                    # 1. Worker Start Constraint
                    for w in workers:
                        links_from_w = {link for link in current_job_links if link[0] == w}
                        for link in links_from_w:
                            tor_switch = link[1]
                            model.addConstr(x[j, link] == 1)
                            model.addConstr(y[j, tor_switch] == 1)

                    # 2. PS End Constraint
                    model.addConstr(x[j, (ps_leaf, ps)] == 1)
                    model.addConstr(y[j, ps_leaf] == 1)

                    # 2. Flow Conservation/Output
                    for node in network.switches:
                        # input_flow -> y[j, node] and a[j, node]
                        links_to_node = {link for link in current_job_links if link[1] == node}
                        if links_to_node:
                            in_flow_sum_j = gp.quicksum(x[j, link] for link in links_to_node)
                            model.addConstr(in_flow_sum_j >= y[j, node])
                            model.addConstr(in_flow_sum_j <= M * y[j, node])

                            # aggregation definition
                            model.addConstr(in_flow_sum_j >= 2 * a[j, node])
                            model.addConstr(in_flow_sum_j <= 1 + M * a[j, node])

                        # output_flow = y[j, node]
                        links_from_node = {link for link in current_job_links if link[0] == node}
                        if links_from_node:
                            out_flow_sum_j = gp.quicksum(x[j, link] for link in links_from_node)
                            model.addConstr(out_flow_sum_j == y[j, node])

                # --- Shared Resource Constraints (FAIR SHARING MODEL) ---

                # 5. Fair Link Bandwidth Constraint
                # If a link is used by n jobs, the throughput of each of these jobs is limited by link_bandwidth / n.
                for j in job_ids:
                    for link in all_relevant_links:
                        # Skip if link has infinite capacity or job j cannot use this link
                        link_bw = network.link_bandwidths.get(link)
                        if link_bw is None or link_bw == float('inf'):
                            continue
                        if (j, link) not in x:
                            continue # x[j, link] does not exist

                        # Number of jobs using this link
                        job_num_link = gp.quicksum(x[k, link] for k in job_ids if (k, link) in x)

                        # If x[j, link] = 1, then lambda_j * job_num_link <= link_bw.
                        # This is linearized using a big-M formulation, active only when x[j, link] is 1.
                        model.addConstr(lambda_j[j] * job_num_link <= link_bw + M_fair_share * (1 - x[j, link]),
                                        name=f"fair_link_bw_{j}_{link[0]}_{link[1]}")

                # 6. Fair Switch Aggregation Capacity Constraint
                # If a switch is used for aggregation by n jobs, the throughput of each is limited by switch_capacity / n.
                for j in job_ids:
                    for node in network.switches:
                        switch_cap = network.switch_processing_capacities.get(node)
                        if switch_cap is None:
                            continue # Should not happen if network is well-defined
                        if (j, node) not in a:
                            continue # a[j, node] does not exist

                        # Number of jobs aggregating at this switch
                        job_num_agg = gp.quicksum(a[k, node] for k in job_ids if (k, node) in a)

                        # If a[j, node] = 1, then lambda_j * job_num_agg <= switch_cap.
                        # Linearized using big-M, active only when a[j, node] is 1.
                        model.addConstr(lambda_j[j] * job_num_agg <= switch_cap + M_fair_share * (1 - a[j, node]),
                                        name=f"fair_switch_cap_{j}_{node}")

                # --- Solve ---
                model.optimize()

                # --- Extract Results ---
                solve_time = time.time() - start_time
                solution_details['solve_time'] = solve_time
                solution_details['model_status'] = model.Status

                if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
                    solution_details['total_max_throughput'] = model.ObjVal
                    for j in job_ids:
                        job_result = {}
                        job_result['id'] = j
                        job_result['throughput'] = lambda_j[j].X
                        job_specific_link_set = job_specific_links.get(j, set())
                        job_result['x'] = {link: x[j, link].X for link in job_specific_link_set if (j, link) in x}
                        job_result['y'] = {node: y[j, node].X for node in network.switches if (j,node) in y} # Check index exists
                        job_result['a'] = {node: a[j, node].X for node in network.switches if (j,node) in a} # Check index exists
                        solution_details['jobs'][j] = job_result
                elif model.Status == GRB.INFEASIBLE:
                    print("Model is infeasible. No solution exists.")
                else:
                    print(f"Optimization finished with status: {model.Status}")
    except gp.GurobiError as e:
        print(f'A Gurobi error occurred: {e}')
        solution_details['model_status'] = f'GurobiError: {e}'
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        solution_details['model_status'] = f'Exception: {e}'

    return solution_details

# For a directed graph with node weights, find the widest path between two points. 
# The width of a path is defined as min(weight of each edge, weight of each node).
def dijkstra_score_search(network_score: Dict, worker: str, ps: str):
    """
    Use Dijkstra's algorithm to find the widest path from worker to ps.
    Args:
        network_score: Dict, network score matrix, network_score[u][v] represents the weight of the edge from u to v.
        worker: str, worker node.
        ps: str, parameter server node.
    Returns:
        path: List[str], the widest path from worker to ps.
        path_score: float, the width score of the path.
    """
    def get_path_width(path: List[str]) -> float:
        """Calculate the width of a given path."""
        if not path:
            return 0
        # Calculate the minimum of all edge weights and all node weights on the path
        edge_weights = [network_score[path[i]][path[i+1]] for i in range(len(path)-1)]
        node_weights = [network_score[node][node] for node in path]
        return min(edge_weights + node_weights)

    best_path = []
    best_score = -1.0
    
    # Initialization
    distances = {node: float('-inf') for node in network_score.keys()}  # Use negative infinity as initial distance
    distances[worker] = float('inf')  # The initial distance of the worker is positive infinity
    prev = {node: None for node in network_score.keys()}
    unvisited = set(network_score.keys())
    
    while unvisited:
        # Find the unvisited node with the largest distance
        current = max(unvisited, key=lambda x: distances[x])
        if distances[current] == float('-inf'):
            break  # Unreachable node
            
        unvisited.remove(current)
        
        # If ps is reached, record the path
        if current == ps:
            path = []
            while current is not None:
                path.append(current)
                current = prev[current]
            path.reverse()
            path_score = get_path_width(path)
            if path_score > best_score:
                best_path = path
                best_score = path_score
            break
        
        # Update the distance of neighboring nodes
        for neighbor in network_score[current]:
            if neighbor in unvisited and network_score[current][neighbor] > 0:
                # Calculate the new path width through the current node
                new_score = min(distances[current], 
                              network_score[current][neighbor],
                              network_score[neighbor][neighbor])
                if new_score > distances[neighbor]:
                    distances[neighbor] = new_score
                    prev[neighbor] = current

    return best_path, best_score


def score_search(network: SpineLeafNetwork, jobs: List[Dict], lp_solution: Dict):
    # Job order, sort jobs by throughput in descending order.
    job_order = sorted(lp_solution['job_routing_result'], key=lambda x: x['throughput'], reverse=True)
    
    jobs_routing_result = {job['id']: {'selected_links': set(), 'active_switches': set(), 'aggregation_switches': set()} for job in jobs}
    jobs_map = {job['id']: job for job in jobs}
    ps_leaf_map = {job['id']: job['ps_leaf'] for job in jobs}
    job_links = {job['id']: set() for job in jobs}

    # Get link_to_job_num_map
    link_to_job_num_map = {}
    for link in network.potential_links:
        link_to_job_num_map[link] = 0
    
    all_nodes = network.nodes
    
    # Find path for each job
    for cur_job_data_from_lp in job_order:
        job_id = cur_job_data_from_lp['id']
        job_selected_links = set()
        job_active_switches = set()
        workers = jobs_map[job_id]['workers']
        ps = jobs_map[job_id]['ps']

        network_score = {u: {v: 0 for v in all_nodes} for u in all_nodes}
        for node in all_nodes:
            network_score[node][node] = float('inf')

        # Assign values to network_score based on cur_job_data_from_lp
        for link in cur_job_data_from_lp['links']:
            link_tuple = (link['source'], link['target'])
            link_bw = network.link_bandwidths.get(link_tuple)
            link_prob = link['prob']
            network_score[link_tuple[0]][link_tuple[1]] = (link_bw * link_prob) / (2**(link_to_job_num_map.get(link_tuple, 0)))
        for switch in cur_job_data_from_lp['aggregation_switches']:
            switch_cap = network.switch_processing_capacities.get(switch)
            switch_prob = cur_job_data_from_lp['aggregation_switches'][switch]
            if switch_prob == 0:
                # Infinity, indicates that this switch will not be a bottleneck.
                network_score[switch][switch] = float('inf')
            else:
                network_score[switch][switch] = (switch_cap * switch_prob) / (2**(link_to_job_num_map.get(link_tuple, 0)))
      
        for worker in workers:
            best_path, best_score = dijkstra_score_search(network_score, worker, ps)
            if best_path:
                for i in range(len(best_path) - 1):
                    link_tuple = (best_path[i], best_path[i + 1])
                    job_selected_links.add(link_tuple)
                    link_to_job_num_map[link_tuple] += 1
        
        job_links[job_id] = job_selected_links
        jobs_routing_result[job_id]['selected_links'] = job_selected_links
        jobs_routing_result[job_id]['active_switches'] = job_active_switches
    
    # Find the aggregation switch for each job based on the job's routing path
    # A switch with an in-degree of 2 or more is considered an aggregation switch
    for job_id in jobs_routing_result:
        job_selected_links = jobs_routing_result[job_id]['selected_links']
        job_active_switches = jobs_routing_result[job_id]['active_switches']
        job_aggregation_switches = set()
                
        for switch in job_active_switches:
            count = 0
            for link in job_selected_links:
                if link[1] == switch:
                    count += 1
                if count >= 2:
                    job_aggregation_switches.add(switch)
                    break
        jobs_routing_result[job_id]['aggregation_switches'] = job_aggregation_switches
    
    return job_links, jobs_routing_result


def convert_lp_to_int_solution_score(network: SpineLeafNetwork, jobs: List[Dict], lp_solution: Dict):
    # Job order, sort jobs by throughput in descending order.
    job_order = sorted(lp_solution['job_routing_result'], key=lambda x: x['throughput'], reverse=True)
    
    jobs_routing_result = {job['id']: {'selected_links': set(), 'active_switches': set(), 'aggregation_switches': set()} for job in jobs}
    jobs_map = {job['id']: job for job in jobs}
    ps_leaf_map = {job['id']: job['ps_leaf'] for job in jobs}
    job_links = {job['id']: set() for job in jobs}

    # Get link_to_job_num_map
    link_to_job_num_map = {}
    for link in network.potential_links:
        link_to_job_num_map[link] = 0

    # Find path for each job
    for cur_job_data_from_lp in job_order:
        job_id = cur_job_data_from_lp['id']
        job_selected_links = set()
        job_active_switches = set()

        # Find the links from workers to leaf_switches, which are fixed.
        workers = jobs_map[job_id]['workers']
        worker_leaf_switches = set()

        for worker in workers:
            for link in cur_job_data_from_lp['links']:
                if worker == link['source']:
                    link_tuple = (worker, link['target'])
                    worker_leaf_switches.add(link['target'])
                    job_selected_links.add(link_tuple)
                    link_to_job_num_map[link_tuple] += 1

        # Find the links from worker_leaf_switch to spine switch.
        leaf_spine_switches = set()
        for wl_switch in worker_leaf_switches:
            if wl_switch == ps_leaf_map[job_id]:
                continue
            max_score = 0
            select_spine_switch = 'None'
            for link in cur_job_data_from_lp['links']:
                if link['source'] == wl_switch:
                    link_tuple = (link['source'], link['target'])
                    #link_prob = link['prob'] / (2**(link_to_job_num_map.get(link_tuple, 0)))
                    #link_prob = link['prob']
                    # Define the score of each link = min(link_bw * link_prob, aggregation_switch_cap * switch_prob)
                    link_bw = network.link_bandwidths.get(link_tuple)
                    switch_cap = network.switch_processing_capacities.get(link['target'])
                    switch_prob = cur_job_data_from_lp['aggregation_switches'][link['target']]
                    if switch_prob == 0:
                        link_score = link_bw * link['prob']
                    else:
                        link_score = min(link_bw * link['prob'], switch_cap * switch_prob)
                    link_score = link_score / (2**(link_to_job_num_map.get(link_tuple, 0)))
                    if link_score > max_score:
                        select_spine_switch = link['target']
                        max_score = link_score
            if select_spine_switch == 'None':
                print("Error in convert_lp_to_int_solution Function!")
            else:
                leaf_spine_switches.add(select_spine_switch)
                link_tuple = (wl_switch, select_spine_switch)
                job_selected_links.add(link_tuple)
                link_to_job_num_map[link_tuple] += 1
        
        # Determine the link from spine_switch to ps_leaf.
        ps_leaf = jobs_map[job_id]['ps_leaf']
        for spine_switch in leaf_spine_switches:
            link_tuple = (spine_switch, ps_leaf)
            job_selected_links.add(link_tuple)
            link_to_job_num_map[link_tuple] += 1
        
        # Get active_switches
        job_active_switches = worker_leaf_switches | leaf_spine_switches
        job_active_switches.add(ps_leaf)

        job_links[job_id] = job_selected_links
        jobs_routing_result[job_id]['selected_links'] = job_selected_links
        jobs_routing_result[job_id]['active_switches'] = job_active_switches

    # Find the aggregation switch for each job based on the job's routing path
    # A switch with an in-degree of 2 or more is considered an aggregation switch
    for job_id in jobs_routing_result:
        job_selected_links = jobs_routing_result[job_id]['selected_links']
        job_active_switches = jobs_routing_result[job_id]['active_switches']
        job_aggregation_switches = set()
                
        for switch in job_active_switches:
            count = 0
            for link in job_selected_links:
                if link[1] == switch:
                    count += 1
                if count >= 2:
                    job_aggregation_switches.add(switch)
                    break
        jobs_routing_result[job_id]['aggregation_switches'] = job_aggregation_switches
    
    #print(jobs_routing_result)
    return job_links, jobs_routing_result

def convert_lp_to_int_solution(network: SpineLeafNetwork, jobs: List[Dict], lp_solution: Dict):
    jobs_routing_result = {job['id']: {'selected_links': set(), 'active_switches': set(), 'aggregation_switches': set()} for job in jobs}
    jobs_map = {job['id']: job for job in jobs}
    job_links = {job['id']: set() for job in jobs}

    # Get link_to_job_num_map
    link_to_job_num_map = {}
    for link in network.potential_links:
        link_to_job_num_map[link] = 0

    # Find path for each job
    for cur_job_data_from_lp in lp_solution['job_routing_result']:
        job_id = cur_job_data_from_lp['id']
        job_selected_links = set()
        job_active_switches = set()

        # Find the links from workers to leaf_switches, which are fixed.
        workers = jobs_map[job_id]['workers']
        worker_leaf_switches = set()

        for worker in workers:
            for link in cur_job_data_from_lp['links']:
                if worker == link['source']:
                    link_tuple = (worker, link['target'])
                    worker_leaf_switches.add(link['target'])
                    job_selected_links.add(link_tuple)
                    link_to_job_num_map[link_tuple] += 1

        # Find the links from worker_leaf_switch to spine switch.
        leaf_spine_switches = set()
        for wl_switch in worker_leaf_switches:
            max_prob = 0
            select_spine_switch = 'None'
            for link in cur_job_data_from_lp['links']:
                if link['source'] == wl_switch: 
                    link_tuple = (link['source'], link['target'])
                    link_prob = link['prob'] / (2**(link_to_job_num_map.get(link_tuple, 0)))
                    #link_prob = link['prob']
                    if link_prob > max_prob:
                        select_spine_switch = link['target']
                        max_prob = link_prob
            if select_spine_switch == 'None':
                print("Error in convert_lp_to_int_solution Function!")
            else:
                leaf_spine_switches.add(select_spine_switch)
                link_tuple = (wl_switch, select_spine_switch)
                job_selected_links.add(link_tuple)
                link_to_job_num_map[link_tuple] += 1
        
        # Determine the link from spine_switch to ps_leaf.
        ps_leaf = jobs_map[job_id]['ps_leaf']
        for spine_switch in leaf_spine_switches:
            link_tuple = (spine_switch, ps_leaf)
            job_selected_links.add(link_tuple)
            link_to_job_num_map[link_tuple] += 1
        
        # Get active_switches
        job_active_switches = worker_leaf_switches | leaf_spine_switches
        job_active_switches.add(ps_leaf)

        job_links[job_id] = job_selected_links
        jobs_routing_result[job_id]['selected_links'] = job_selected_links
        jobs_routing_result[job_id]['active_switches'] = job_active_switches

    # Find the aggregation switch for each job based on the job's routing path
    # A switch with an in-degree of 2 or more is considered an aggregation switch
    for job_id in jobs_routing_result:
        job_selected_links = jobs_routing_result[job_id]['selected_links']
        job_active_switches = jobs_routing_result[job_id]['active_switches']
        job_aggregation_switches = set()
                
        for switch in job_active_switches:
            count = 0
            for link in job_selected_links:
                if link[1] == switch:
                    count += 1
                if count >= 2:
                    job_aggregation_switches.add(switch)
                    break
        jobs_routing_result[job_id]['aggregation_switches'] = job_aggregation_switches
    
    #print(jobs_routing_result)
    return job_links, jobs_routing_result

def get_lp_solution(solution_details, verbose=False, output_json=False, output_filepath=None):
    solve_time = solution_details.get('solve_time', 0)
    total_throughput = solution_details.get('total_max_throughput', 0.0)

    jobs_selected_links = {}
    jobs_active_switches = {}
    jobs_aggregation_switches = {}
    
    for job_id, job_details in solution_details['jobs'].items():
        jobs_selected_links[job_id] = job_details['x']
        jobs_active_switches[job_id] = job_details['y']
        jobs_aggregation_switches[job_id] = job_details['a']
    
    #print(f"Model Status: {solution_details.get('model_status', 'N/A')}")
    #print(f"Solve Time: {solve_time:.3f} seconds")
    #print(f"Maximum Sum Throughput: {total_throughput:.4f}")

    if verbose and solution_details.get('jobs'):
        print(" Job Details:")
        for j in solution_details['jobs']:
            print(f"  Job {solution_details['jobs'][j]['id']}:")
            print(f"    Throughput: {solution_details['jobs'][j]['throughput']:.4f}")
            print(f"    Selected Links(x): {solution_details['jobs'][j]['x']}")
            print(f"    Active Switches(y): {solution_details['jobs'][j]['y']}")
            print(f"    Aggregation Switches(a): {solution_details['jobs'][j]['a']}")

    # Output to JSON file if requested
    if output_json and output_filepath:     
        output_data = {
            "model_status": solution_details.get('model_status', 'N/A'),
            "solve_time": solve_time,
            "maximum_sum_throughput": total_throughput,
            "job_routing_result": []
        }
        if solution_details.get('jobs'):
            for j in solution_details['jobs']:
                job_data = {
                    "id": solution_details['jobs'][j]['id'],
                    "throughput": round(solution_details['jobs'][j]['throughput'], 4),
                    "links":[
                        {"source": link_tuple[0], "target": link_tuple[1], "prob": prob_val} 
                        for link_tuple, prob_val in jobs_selected_links[j].items()
                    ],
                    "active_switches": { 
                        active_switch_node : prob_val 
                        for active_switch_node, prob_val in jobs_active_switches[j].items()
                    },
                    "aggregation_switches": {
                        aggregation_switch_node : prob_val 
                        for aggregation_switch_node, prob_val in jobs_aggregation_switches[j].items()
                    }
                }
                output_data["job_routing_result"].append(job_data)

        # Write formatted data to JSON file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, sort_keys=False)
    lp_solution = output_data
    return lp_solution

def get_int_solution_json_file(jobs_routing_result, output_json=False, output_filepath=None):
    output_data = {}
    for job_id in jobs_routing_result:
        output_data[job_id] = {
            'selected_links': sorted(list(jobs_routing_result[job_id]['selected_links'])),
            'active_switches': sorted(list(jobs_routing_result[job_id]['active_switches'])),
            'aggregation_switches': sorted(list(jobs_routing_result[job_id]['aggregation_switches']))
        }
    if output_json and output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, sort_keys=False)

def generate_job_links_json(job_links, output_filepath):
    serializable_job_links = {}
    for job_id, links in job_links.items():
        # Convert each set of links to a sorted list of tuples
        serializable_job_links[str(int(job_id))] = sorted(list(links))

    try:
        with open(output_filepath, 'w') as f:
            json.dump(serializable_job_links, f, indent=4, sort_keys=True)
        print(f"\nSuccessfully wrote job links to: {output_filepath}")
    except IOError as e:
        print(f"\nError writing to file {output_filepath}: {e}")
    except TypeError as e:
        print(f"\nError serializing data to JSON: {e}. Check data structures in job_links.")