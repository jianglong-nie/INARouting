import gurobipy as gp
from gurobipy import GRB
import time
import json
from typing import Optional, Dict, List
from network_generator.fat_tree_network import FatTreeNetwork


def _get_pod_from_name(switch_name):
    """Helper to extract pod number from a switch name like 'pod1_edge2'."""
    if not isinstance(switch_name, str) or 'pod' not in switch_name:
        return None
    try:
        return int(switch_name.split('_')[0].replace('pod', ''))
    except (ValueError, IndexError):
        return None

# --- Multi-Job Solver Function ---
def solve_fat_tree_opt(network: FatTreeNetwork,
                   jobs: List[Dict],
                   gurobi_params: Optional[dict] = None):
    """
    Solves the multi-job max sum-throughput problem for FatTree using Gurobi.
    """
    solution_details = {'model_status': None, 'solve_time': None,
                        'total_max_throughput': 0.0, 'jobs': {}}
    start_time = time.time()

    job_ids = [job['id'] for job in jobs]
    job_map = {job['id']: job for job in jobs}

    job_specific_links = {
        job['id']: network.get_job_specific_links(job) for job in jobs
    }
    valid_x_indices = [
        (j, link)
        for j in job_ids
        for link in job_specific_links.get(j, set())
    ]
    all_relevant_links = set().union(*job_specific_links.values())

    try:
        with gp.Env(params=gurobi_params) as env:
            with gp.Model(env=env, name="FatTreeMaxSumThroughput") as model:
                x = model.addVars(valid_x_indices, vtype=GRB.BINARY, name="x")
                y = model.addVars(job_ids, network.switches, vtype=GRB.BINARY, name="y")
                a = model.addVars(job_ids, network.switches, vtype=GRB.BINARY, name="a")
                lambda_j = model.addVars(job_ids, vtype=GRB.CONTINUOUS, lb=0, name="lambda")

                model.setObjective(gp.quicksum(lambda_j[j] for j in job_ids), GRB.MAXIMIZE)

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
                    ps_edge = job['ps_edge']
                    current_job_links = job_specific_links[j]

                    for w in workers:
                        links_from_w = {link for link in current_job_links if link[0] == w}
                        for link in links_from_w:
                            tor_switch = link[1]
                            model.addConstr(x[j, link] == 1, name=f"worker_start_{j}_{w}_{tor_switch}")
                            model.addConstr(y[j, tor_switch] == 1, name=f"y_worker_start_{j}_{w}_{tor_switch}")

                    model.addConstr(x[j, (ps_edge, ps)] == 1, name=f"ps_end_{j}")
                    model.addConstr(y[j, ps_edge] == 1, name=f"y_ps_end_{j}")

                    for node in network.switches:
                        links_to_node = {link for link in current_job_links if link[1] == node}
                        if links_to_node:
                            in_flow_sum_j = gp.quicksum(x[j, link] for link in links_to_node)
                            model.addConstr(in_flow_sum_j >= y[j, node], name=f"inflow_ge_y_{j}_{node}")
                            model.addConstr(in_flow_sum_j <= M * y[j, node], name=f"inflow_le_My_{j}_{node}")
                            model.addConstr(in_flow_sum_j >= 2 * a[j, node], name=f"agg_def_ge_{j}_{node}")
                            model.addConstr(in_flow_sum_j <= 1 + M * a[j, node], name=f"agg_def_le_{j}_{node}")

                        links_from_node = {link for link in current_job_links if link[0] == node}
                        if links_from_node:
                            out_flow_sum_j = gp.quicksum(x[j, link] for link in links_from_node)
                            model.addConstr(out_flow_sum_j == y[j, node], name=f"flow_conservation_{j}_{node}")
                
                
                # --- Shared Resource Constraints (FAIR SHARING MODEL) ---
                for j in job_ids:
                    for link in job_specific_links.get(j, set()):
                        link_bw = network.link_bandwidths.get(link)
                        if link_bw is None or link_bw == float('inf'):
                            continue
                        job_num_link = gp.quicksum(x[k, link] for k in job_ids if (k, link) in x)
                        model.addConstr(lambda_j[j] * job_num_link <= link_bw + M_fair_share * (1 - x[j, link]),
                                        name=f"fair_link_bw_{j}_{link[0]}_{link[1]}")

                    for node in network.switches:
                        switch_cap = network.switch_processing_capacities.get(node)
                        if switch_cap is None:
                            continue
                        job_num_switch = gp.quicksum(a[k, node] for k in job_ids if (k, node) in a)
                        model.addConstr(lambda_j[j] * job_num_switch <= switch_cap + M_fair_share * (1 - a[j, node]),
                                        name=f"fair_cap_switch_{j}_{node}")

                model.optimize()

                solve_time = time.time() - start_time
                solution_details['solve_time'] = solve_time
                solution_details['model_status'] = model.Status

                if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    solution_details['total_max_throughput'] = model.ObjVal
                    for j in job_ids:
                        job_result = {
                            'id': j,
                            'throughput': lambda_j[j].X,
                            'x': {link: x[j, link].X for link in job_specific_links.get(j, set()) if (j, link) in x},
                            'y': {node: y[j, node].X for node in network.switches if (j, node) in y},
                            'a': {node: a[j, node].X for node in network.switches if (j, node) in a}
                        }
                        solution_details['jobs'][j] = job_result
                elif model.Status == GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS to find conflicting constraints.")
                    model.computeIIS()
                    model.write("fat_tree_model.ilp")
                    print("IIS written to fat_tree_model.ilp")
                else:
                    print(f"Optimization finished with status: {model.Status}")

    except gp.GurobiError as e:
        print(f'A Gurobi error occurred: {e}')
        solution_details['model_status'] = f'GurobiError: {e}'
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        solution_details['model_status'] = f'Exception: {e}'

    return solution_details


# --- Multi-Job Solver Function ---
def solve_fat_tree(network: FatTreeNetwork,
                   jobs: List[Dict],
                   gurobi_params: Optional[dict] = None):
    """
    Solves the multi-job max sum-throughput problem for FatTree using Gurobi.
    """
    solution_details = {'model_status': None, 'solve_time': None,
                        'total_max_throughput': 0.0, 'jobs': {}}
    start_time = time.time()

    job_ids = [job['id'] for job in jobs]
    job_map = {job['id']: job for job in jobs}

    job_specific_links = {
        job['id']: network.get_job_specific_links(job) for job in jobs
    }
    valid_x_indices = [
        (j, link)
        for j in job_ids
        for link in job_specific_links.get(j, set())
    ]
    all_relevant_links = set().union(*job_specific_links.values())

    try:
        with gp.Env(params=gurobi_params) as env:
            with gp.Model(env=env, name="FatTreeMaxSumThroughput") as model:
                x = model.addVars(valid_x_indices, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
                y = model.addVars(job_ids, network.switches, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
                a = model.addVars(job_ids, network.switches, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="a")
                lambda_j = model.addVars(job_ids, vtype=GRB.CONTINUOUS, lb=0, name="lambda")

                model.setObjective(gp.quicksum(lambda_j[j] for j in job_ids), GRB.MAXIMIZE)

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
                    ps_edge = job['ps_edge']
                    current_job_links = job_specific_links[j]

                    for w in workers:
                        links_from_w = {link for link in current_job_links if link[0] == w}
                        for link in current_job_links:
                            if link in links_from_w:
                                tor_switch = link[1]
                                model.addConstr(x[j, link] == 1, name=f"worker_start_{j}_{w}_{tor_switch}")
                                model.addConstr(y[j, tor_switch] == 1, name=f"y_worker_start_{j}_{w}_{tor_switch}")

                    model.addConstr(x[j, (ps_edge, ps)] == 1, name=f"ps_end_{j}")
                    model.addConstr(y[j, ps_edge] == 1, name=f"y_ps_end_{j}")

                    for node in network.switches:
                        links_to_node = {link for link in current_job_links if link[1] == node}
                        if links_to_node:
                            in_flow_sum_j = gp.quicksum(x[j, link] for link in links_to_node)
                            model.addConstr(in_flow_sum_j >= y[j, node], name=f"inflow_ge_y_{j}_{node}")
                            model.addConstr(in_flow_sum_j <= M * y[j, node], name=f"inflow_le_My_{j}_{node}")
                            model.addConstr(in_flow_sum_j >= 2 * a[j, node], name=f"agg_def_ge_{j}_{node}")
                            model.addConstr(in_flow_sum_j <= 1 + M * a[j, node], name=f"agg_def_le_{j}_{node}")

                        links_from_node = {link for link in current_job_links if link[0] == node}
                        if links_from_node:
                            out_flow_sum_j = gp.quicksum(x[j, link] for link in links_from_node)
                            model.addConstr(out_flow_sum_j == y[j, node], name=f"flow_conservation_{j}_{node}")
                
                # --- Fair Sharing Constraints ---
                for j in job_ids:
                    for link in job_specific_links.get(j, set()):
                        link_bw = network.link_bandwidths.get(link)
                        if link_bw is None or link_bw == float('inf'):
                            continue
                        job_num_link = gp.quicksum(x[k, link] for k in job_ids if (k, link) in x)
                        model.addConstr(lambda_j[j] * job_num_link <= link_bw + M_fair_share * (1 - x[j, link]),
                                        name=f"fair_link_bw_{j}_{link[0]}_{link[1]}")

                    for node in network.switches:
                        switch_cap = network.switch_processing_capacities.get(node)
                        if switch_cap is None:
                            continue
                        job_num_switch = gp.quicksum(a[k, node] for k in job_ids if (k, node) in a)
                        model.addConstr(lambda_j[j] * job_num_switch <= switch_cap + M_fair_share * (1 - a[j, node]),
                                        name=f"fair_cap_switch_{j}_{node}")

                model.optimize()

                solve_time = time.time() - start_time
                solution_details['solve_time'] = solve_time
                solution_details['model_status'] = model.Status

                if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    solution_details['total_max_throughput'] = model.ObjVal
                    for j in job_ids:
                        job_result = {
                            'id': j,
                            'throughput': lambda_j[j].X,
                            'x': {link: x[j, link].X for link in job_specific_links.get(j, set()) if (j, link) in x},
                            'y': {node: y[j, node].X for node in network.switches if (j, node) in y},
                            'a': {node: a[j, node].X for node in network.switches if (j, node) in a}
                        }
                        solution_details['jobs'][j] = job_result
                elif model.Status == GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS to find conflicting constraints.")
                    model.computeIIS()
                    model.write("fat_tree_model.ilp")
                    print("IIS written to fat_tree_model.ilp")
                else:
                    print(f"Optimization finished with status: {model.Status}")

    except gp.GurobiError as e:
        print(f'A Gurobi error occurred: {e}')
        solution_details['model_status'] = f'GurobiError: {e}'
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        solution_details['model_status'] = f'Exception: {e}'

    return solution_details


def dijkstra_score_search(network_score: Dict, worker: str, ps: str):
    """
    Use Dijkstra's algorithm to find the widest path from worker to ps.
    Args:
       
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
        # Find the node with the maximum distance among unvisited nodes
        current = max(unvisited, key=lambda x: distances[x])
        if distances[current] == float('-inf'):
            break  # Unreachable nodes
            
        unvisited.remove(current)
        
        # If the current node is ps, record the path
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
            if neighbor in unvisited and network_score[current].get(neighbor, 0) > 0:
                # Calculate the new path width through the current node
                new_score = min(distances[current], 
                              network_score[current][neighbor],
                              network_score[neighbor][neighbor])
                if new_score > distances[neighbor]:
                    distances[neighbor] = new_score
                    prev[neighbor] = current

    return best_path, best_score


def score_search(network: FatTreeNetwork, jobs: List[Dict], lp_solution: Dict):
    # Job order, sort jobs by throughput in descending order.
    job_order = sorted(lp_solution.get('job_routing_result', []), key=lambda x: x['throughput'], reverse=True)
    
    jobs_routing_result = {job['id']: {'selected_links': set(), 'active_switches': set(), 'aggregation_switches': set()} for job in jobs}
    jobs_map = {job['id']: job for job in jobs}
    job_links = {job['id']: set() for job in jobs}

    # Get link_to_job_num_map and switch_to_job_num_map
    link_to_job_num_map = {link: 0 for link in network.potential_links}
    switch_to_job_num_map = {switch: 0 for switch in network.switches}

    all_nodes = network.nodes
    
    # Find path for each job
    for cur_job_data_from_lp in job_order:
        job_id = cur_job_data_from_lp['id']
        job_selected_links = set()
        workers = jobs_map[job_id]['workers']
        ps = jobs_map[job_id]['ps']

        network_score = {u: {v: 0 for v in all_nodes} for u in all_nodes}
        for node in all_nodes:
            network_score[node][node] = float('inf')

        # Assign values to network_score based on cur_job_data_from_lp
        for link in cur_job_data_from_lp.get('links', []):
            link_tuple = (link['source'], link['target'])
            if link_tuple not in network.link_bandwidths:
                continue
            link_bw = network.link_bandwidths.get(link_tuple)
            link_prob = link['prob']
            # Apply penalty
            network_score[link_tuple[0]][link_tuple[1]] = (link_bw * link_prob) / (2**(link_to_job_num_map.get(link_tuple, 0)))
        
        for switch, switch_prob in cur_job_data_from_lp.get('aggregation_switches', {}).items():
            if switch not in network.switch_processing_capacities:
                continue
            switch_cap = network.switch_processing_capacities.get(switch)
            if switch_prob == 0:
                network_score[switch][switch] = float('inf')
            else:
                # Apply penalty
                network_score[switch][switch] = (switch_cap * switch_prob) / (2**(switch_to_job_num_map.get(switch, 0)))
      
        for worker in workers:
            best_path, best_score = dijkstra_score_search(network_score, worker, ps)
            if best_path:
                for i in range(len(best_path) - 1):
                    link_tuple = (best_path[i], best_path[i + 1])
                    job_selected_links.add(link_tuple)
                    link_to_job_num_map[link_tuple] = link_to_job_num_map.get(link_tuple, 0) + 1
                for node in best_path:
                    if node in network.switches:
                        switch_to_job_num_map[node] = switch_to_job_num_map.get(node, 0) + 1
        
        job_links[job_id] = job_selected_links
        jobs_routing_result[job_id]['selected_links'] = job_selected_links
        
        # Determine active switches for this job
        active_switches = set()
        for u, v in job_selected_links:
            if u in network.switches:
                active_switches.add(u)
            if v in network.switches:
                active_switches.add(v)
        jobs_routing_result[job_id]['active_switches'] = active_switches
        
        # Determine aggregation switches for this job
        aggregation_switches = set()
        for switch in active_switches:
            in_degree = sum(1 for u, v in job_selected_links if v == switch)
            if in_degree >= 2:
                aggregation_switches.add(switch)
        jobs_routing_result[job_id]['aggregation_switches'] = aggregation_switches
    
    return job_links, jobs_routing_result


def convert_lp_to_int_solution(network: FatTreeNetwork, jobs: List[Dict], lp_solution: Dict):
    jobs_routing_result = {job['id']: {'selected_links': set(), 'active_switches': set(), 'aggregation_switches': set()} for job in jobs}
    jobs_map = {job['id']: job for job in jobs}
    job_links = {job['id']: set() for job in jobs}

    link_to_job_num_map = {}
    for link in network.potential_links:
        link_to_job_num_map[link] = 0

    for cur_job_data_from_lp in lp_solution.get('job_routing_result', []):
        job_id = cur_job_data_from_lp['id']
        job_selected_links = set()
        
        workers = jobs_map[job_id]['workers']
        worker_edge_switches = set()

        job_x_links = {
            (link['source'], link['target']): link['prob']
            for link in cur_job_data_from_lp.get('links', [])
        }

        for worker in workers:
            w_edge = network.server_to_edge_map.get(worker)
            if w_edge:
                link_tuple = (worker, w_edge)
                worker_edge_switches.add(w_edge)
                job_selected_links.add(link_tuple)
                link_to_job_num_map.setdefault(link_tuple, 0)
                link_to_job_num_map[link_tuple] += 1

        selected_agg_switches = set()
        for w_edge in worker_edge_switches:
            max_prob = -1
            selected_agg = None
            for (u, v), prob in job_x_links.items():
                if u == w_edge and v in network.agg_switches:
                    score = prob / (2**(link_to_job_num_map.get((u,v), 0)))
                    if score > max_prob:
                        max_prob = score
                        selected_agg = v
            if selected_agg:
                selected_agg_switches.add(selected_agg)
                link_tuple = (w_edge, selected_agg)
                job_selected_links.add(link_tuple)
                link_to_job_num_map.setdefault(link_tuple, 0)
                link_to_job_num_map[link_tuple] += 1

        ps_edge = jobs_map[job_id]['ps_edge']
        ps_pod = _get_pod_from_name(ps_edge)

        if ps_pod is None:
            print(f"Error: Could not determine pod for ps_edge {ps_edge} in job {job_id}")
            continue

        for agg_switch in selected_agg_switches:
            agg_pod = _get_pod_from_name(agg_switch)
            if agg_pod == ps_pod:
                if (agg_switch, ps_edge) in network.potential_links:
                    job_selected_links.add((agg_switch, ps_edge))
                    link_to_job_num_map.setdefault((agg_switch, ps_edge), 0)
                    link_to_job_num_map[(agg_switch, ps_edge)] += 1
            else:
                # Path: agg_switch -> core_switch -> ps_pod_agg_switch -> ps_edge
                best_core, max_prob_to_core = None, -1
                for (u, v), prob in job_x_links.items():
                    if u == agg_switch and v in network.core_switches and prob > max_prob_to_core:
                        max_prob_to_core, best_core = prob, v
                
                if not best_core: continue
                job_selected_links.add((agg_switch, best_core))
                link_to_job_num_map.setdefault((agg_switch, best_core), 0)
                link_to_job_num_map[(agg_switch, best_core)] += 1

                best_ps_agg, max_prob_to_ps_agg = None, -1
                for (u, v), prob in job_x_links.items():
                    if u == best_core and _get_pod_from_name(v) == ps_pod and prob > max_prob_to_ps_agg:
                        max_prob_to_ps_agg, best_ps_agg = prob, v

                if not best_ps_agg: continue
                job_selected_links.add((best_core, best_ps_agg))
                link_to_job_num_map.setdefault((best_core, best_ps_agg), 0)
                link_to_job_num_map[(best_core, best_ps_agg)] += 1

                if (best_ps_agg, ps_edge) in network.potential_links:
                    job_selected_links.add((best_ps_agg, ps_edge))
                    link_to_job_num_map.setdefault((best_ps_agg, ps_edge), 0)
                    link_to_job_num_map[(best_ps_agg, ps_edge)] += 1

        job_links[job_id] = job_selected_links
        
        active_switches = set()
        for u, v in job_selected_links:
            if u in network.switches: active_switches.add(u)
            if v in network.switches: active_switches.add(v)
        
        aggregation_switches = set()
        for switch in active_switches:
            in_degree = sum(1 for u, v in job_selected_links if v == switch)
            if in_degree >= 2:
                aggregation_switches.add(switch)

        jobs_routing_result[job_id]['selected_links'] = job_selected_links
        jobs_routing_result[job_id]['active_switches'] = active_switches
        jobs_routing_result[job_id]['aggregation_switches'] = aggregation_switches

    return job_links, jobs_routing_result


def get_lp_solution(solution_details, verbose=False, output_json=False, output_filepath=None):
    solve_time = solution_details.get('solve_time', 0)
    total_throughput = solution_details.get('total_max_throughput', 0.0)

    output_data = {
        "model_status": solution_details.get('model_status', 'N/A'),
        "solve_time": solve_time,
        "maximum_sum_throughput": total_throughput,
        "job_routing_result": []
    }
    if solution_details.get('jobs'):
        for j_id, j_details in solution_details['jobs'].items():
            job_data = {
                "id": j_details['id'],
                "throughput": round(j_details['throughput'], 4),
                "links": [
                    {"source": link[0], "target": link[1], "prob": prob}
                    for link, prob in j_details['x'].items() if prob > 1e-6
                ],
                "active_switches": {
                    node: prob for node, prob in j_details['y'].items() if prob > 1e-6
                },
                "aggregation_switches": {
                    node: prob for node, prob in j_details['a'].items() if prob > 1e-6
                }
            }
            output_data["job_routing_result"].append(job_data)
    
    if verbose:
        print(json.dumps(output_data, indent=4))

    if output_json and output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        #print(f"LP Solution saved to {output_filepath}")

    return output_data


def get_int_solution_json_file(jobs_routing_result, output_json=False, output_filepath=None):
    output_data = {}
    for job_id, data in jobs_routing_result.items():
        output_data[job_id] = {
            'selected_links': sorted(list(data['selected_links'])),
            'active_switches': sorted(list(data['active_switches'])),
            'aggregation_switches': sorted(list(data['aggregation_switches']))
        }
    if output_json and output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)


def generate_job_links_json(job_links, output_filepath):
    serializable_job_links = {
        str(int(job_id)): sorted(list(links))
        for job_id, links in job_links.items()
    }
    try:
        with open(output_filepath, 'w') as f:
            json.dump(serializable_job_links, f, indent=4, sort_keys=True)
        print(f"\nSuccessfully wrote job links to: {output_filepath}")
    except IOError as e:
        print(f"\nError writing to file {output_filepath}: {e}")