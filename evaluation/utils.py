import json
from collections import defaultdict

# Helper function to load JSON files
def load_json_file(file_path):
    """Loads a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON format.")
        return None

def water_filling_spine_leaf(network_config_data, routing_results):
    """
    Calculates the throughput for each job using a water-filling algorithm,
    based on network configuration and routing results for a spine-leaf topology.
    """
    network_json = network_config_data['network_json']

    # 1. Extract network information
    link_capacities_initial = {}
    for link_info in network_json['links']:
        source = link_info['source']
        target = link_info['target']
        bandwidth = link_info['bandwidth']
        link_capacities_initial[(source, target)] = bandwidth

    switch_processing_capacities_initial = network_json['switch_processing_capacities']
    all_switches = set(network_json['nodes']['leaf_switches']) | set(network_json['nodes']['spine_switches'])

    # 2. Parse routing results to find links and aggregation points for each job
    job_routes = {}
    for job_id_str, links_list in routing_results.items():
        job_id = int(job_id_str)
        job_routes[job_id] = {tuple(link_pair) for link_pair in links_list}

    job_aggregation_points = defaultdict(set)
    for job_id, links_used in job_routes.items():
        incoming_counts = defaultdict(int)
        for _, v_node in links_used:
            if v_node in all_switches:
                incoming_counts[v_node] += 1
        for switch_node, count in incoming_counts.items():
            if count >= 2:
                job_aggregation_points[job_id].add(switch_node)

    # --- Water-Filling Algorithm ---
    final_job_throughputs = {}
    frozen_jobs = set()
    active_job_ids = list(job_routes.keys())

    if not active_job_ids:
        return {}

    # These are the available capacities that will be decremented
    link_capacities = link_capacities_initial.copy()
    switch_capacities = switch_processing_capacities_initial.copy()

    while len(frozen_jobs) < len(active_job_ids):
        unfrozen_job_ids = [job_id for job_id in active_job_ids if job_id not in frozen_jobs]
        if not unfrozen_job_ids:
            break

        # 1. Calculate resource loads from unfrozen jobs
        link_loads = defaultdict(int)
        switch_loads = defaultdict(int)
        for job_id in unfrozen_job_ids:
            for link in job_routes.get(job_id, []):
                if link in link_capacities:
                    link_loads[link] += 1
            for switch in job_aggregation_points.get(job_id, []):
                if switch in switch_capacities:
                    switch_loads[switch] += 1

        # 2. Find the bottleneck resource and rate
        min_rate = float('inf')
        bottleneck_resource = None
        for link, load in link_loads.items():
            if load > 0:
                rate = link_capacities[link] / load
                if rate < min_rate:
                    min_rate = rate
                    bottleneck_resource = ('link', link)
        
        for switch, load in switch_loads.items():
            if load > 0:
                rate = switch_capacities[switch] / load
                if rate < min_rate:
                    min_rate = rate
                    bottleneck_resource = ('switch', switch)

        if min_rate == float('inf') or bottleneck_resource is None:
            for job_id in unfrozen_job_ids:
                final_job_throughputs[job_id] = 0.0
            break

        # 3. Freeze jobs on the bottleneck
        jobs_to_freeze_this_step = set()
        resource_type, resource_name = bottleneck_resource
        
        for job_id in unfrozen_job_ids:
            is_on_bottleneck = False
            if resource_type == 'link' and resource_name in job_routes.get(job_id, []):
                is_on_bottleneck = True
            elif resource_type == 'switch' and resource_name in job_aggregation_points.get(job_id, []):
                is_on_bottleneck = True
            
            if is_on_bottleneck:
                final_job_throughputs[job_id] = min_rate
                jobs_to_freeze_this_step.add(job_id)

        if not jobs_to_freeze_this_step:
            break

        # 4. Update resource capacities
        for job_id in jobs_to_freeze_this_step:
            rate = final_job_throughputs[job_id]
            for link in job_routes.get(job_id, []):
                if link in link_capacities:
                    link_capacities[link] -= rate
            for switch in job_aggregation_points.get(job_id, []):
                if switch in switch_capacities:
                    switch_capacities[switch] -= rate
        
        frozen_jobs.update(jobs_to_freeze_this_step)

    # Ensure all jobs have a throughput value
    for job_id in active_job_ids:
        if job_id not in final_job_throughputs:
            final_job_throughputs[job_id] = 0.0

    return final_job_throughputs

def water_filling_fat_tree(network_config_data, routing_results):
    """
    Calculates the throughput for each job using a water-filling algorithm,
    based on network configuration and routing results for a fat-tree topology.
    """
    network_json = network_config_data['network_json']

    # 1. Extract network information for Fat-Tree
    link_capacities_initial = {}
    for link_info in network_json['links']['potential_links_with_bandwidth']:
        source = link_info['source']
        target = link_info['target']
        bandwidth = link_info['bandwidth']
        link_capacities_initial[(source, target)] = bandwidth

    switch_processing_capacities_initial = network_json['switch_processing_capacities']
    all_switches = set(network_json['nodes']['core_switches']) | \
                   set(network_json['nodes']['agg_switches']) | \
                   set(network_json['nodes']['edge_switches'])

    # 2. Parse routing results to find links and aggregation points for each job
    job_routes = {}
    for job_id_str, links_list in routing_results.items():
        job_id = int(job_id_str)
        job_routes[job_id] = {tuple(link_pair) for link_pair in links_list}

    job_aggregation_points = defaultdict(set)
    for job_id, links_used in job_routes.items():
        incoming_counts = defaultdict(int)
        for _, v_node in links_used:
            if v_node in all_switches:
                incoming_counts[v_node] += 1
        for switch_node, count in incoming_counts.items():
            if count >= 2:
                job_aggregation_points[job_id].add(switch_node)

    # --- Water-Filling Algorithm ---
    final_job_throughputs = {}
    frozen_jobs = set()
    active_job_ids = list(job_routes.keys())

    if not active_job_ids:
        return {}

    # These are the available capacities that will be decremented
    link_capacities = link_capacities_initial.copy()
    switch_capacities = switch_processing_capacities_initial.copy()

    while len(frozen_jobs) < len(active_job_ids):
        unfrozen_job_ids = [job_id for job_id in active_job_ids if job_id not in frozen_jobs]
        if not unfrozen_job_ids:
            break

        # 1. Calculate resource loads from unfrozen jobs
        link_loads = defaultdict(int)
        switch_loads = defaultdict(int)
        for job_id in unfrozen_job_ids:
            for link in job_routes.get(job_id, []):
                if link in link_capacities:
                    link_loads[link] += 1
            for switch in job_aggregation_points.get(job_id, []):
                if switch in switch_capacities:
                    switch_loads[switch] += 1

        # 2. Find the bottleneck resource and rate
        min_rate = float('inf')
        bottleneck_resource = None
        for link, load in link_loads.items():
            if load > 0 and link_capacities.get(link, 0) > 0:
                rate = link_capacities[link] / load
                if rate < min_rate:
                    min_rate = rate
                    bottleneck_resource = ('link', link)
        
        for switch, load in switch_loads.items():
            if load > 0 and switch_capacities.get(switch, 0) > 0:
                rate = switch_capacities[switch] / load
                if rate < min_rate:
                    min_rate = rate
                    bottleneck_resource = ('switch', switch)

        if min_rate == float('inf') or bottleneck_resource is None:
            for job_id in unfrozen_job_ids:
                final_job_throughputs[job_id] = 0.0
            break

        # 3. Freeze jobs on the bottleneck
        jobs_to_freeze_this_step = set()
        resource_type, resource_name = bottleneck_resource
        
        for job_id in unfrozen_job_ids:
            is_on_bottleneck = False
            if resource_type == 'link' and resource_name in job_routes.get(job_id, []):
                is_on_bottleneck = True
            elif resource_type == 'switch' and resource_name in job_aggregation_points.get(job_id, []):
                is_on_bottleneck = True
            
            if is_on_bottleneck:
                final_job_throughputs[job_id] = min_rate
                jobs_to_freeze_this_step.add(job_id)

        if not jobs_to_freeze_this_step:
            # This case can happen if min_rate is found but no jobs are on that specific bottleneck,
            # which might indicate an issue in logic or data. Break to be safe.
            for job_id in unfrozen_job_ids:
                if job_id not in final_job_throughputs:
                    final_job_throughputs[job_id] = 0.0
            break

        # 4. Update resource capacities
        for job_id in jobs_to_freeze_this_step:
            rate = final_job_throughputs[job_id]
            for link in job_routes.get(job_id, []):
                if link in link_capacities:
                    link_capacities[link] -= rate
            for switch in job_aggregation_points.get(job_id, []):
                if switch in switch_capacities:
                    switch_capacities[switch] -= rate
        
        frozen_jobs.update(jobs_to_freeze_this_step)

    # Ensure all jobs have a throughput value assigned
    for job_id in active_job_ids:
        if job_id not in final_job_throughputs:
            final_job_throughputs[job_id] = 0.0

    return final_job_throughputs
