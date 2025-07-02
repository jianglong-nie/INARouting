import os
import json
import random
from typing import Optional, List, Dict, Set, Tuple

# --- FatTree Network Topology Class ---
class FatTreeNetwork:
    """
    Represents a K-ary Fat-Tree network topology with Servers.
    Stores nodes (core, aggregation, edge switches, servers),
    directed links, link bandwidths, and switch processing capacities.
    """

    def __init__(self, k: Optional[int] = None, servers_per_edge: Optional[int] = None, random_seed: Optional[int] = None, network_json: Optional[Dict] = None):
        """
        Initializes a K-ary Fat-Tree network.
        Can be initialized either from parameters (k, servers_per_edge) or from a network_json object.

        Args:
            k: The arity of the Fat-Tree. Must be a positive even number.
            servers_per_edge: Number of servers connected to each edge switch.
            random_seed: Seed for random number generation.
            network_json: A dictionary containing the pre-generated network topology.
        """
        if network_json:
            self.read_fat_tree_json_file(network_json)
        elif k is not None and servers_per_edge is not None:
            if k <= 0 or k % 2 != 0:
                raise ValueError("k (arity) must be a positive even integer.")
            if servers_per_edge <= 0:
                raise ValueError("servers_per_edge must be positive.")
            
            self.rng = random.Random(random_seed)

            self.k = k
            self.servers_per_edge = servers_per_edge

            self.num_pods = k
            self.num_core_switches = (k // 2)**2
            self.num_agg_switches_per_pod = k // 2
            self.num_edge_switches_per_pod = k // 2

            # Node Creation
            self.core_switches = [f'core{i}' for i in range(1, self.num_core_switches + 1)]
            self.agg_switches = []
            self.edge_switches = []

            self.pod_to_agg_switches: Dict[int, Set[str]] = {p: set() for p in range(1, self.num_pods + 1)}
            self.pod_to_edge_switches: Dict[int, Set[str]] = {p: set() for p in range(1, self.num_pods + 1)}

            for p_idx in range(1, self.num_pods + 1):
                for s_idx in range(1, self.num_agg_switches_per_pod + 1):
                    agg_name = f'pod{p_idx}_agg{s_idx}'
                    self.agg_switches.append(agg_name)
                    self.pod_to_agg_switches[p_idx].add(agg_name)
                for s_idx in range(1, self.num_edge_switches_per_pod + 1):
                    edge_name = f'pod{p_idx}_edge{s_idx}'
                    self.edge_switches.append(edge_name)
                    self.pod_to_edge_switches[p_idx].add(edge_name)

            self.switches = self.core_switches + self.agg_switches + self.edge_switches

            self.all_servers = []
            self.server_to_edge_map: Dict[str, str] = {}
            self.edge_to_servers_map: Dict[str, Set[str]] = {edge: set() for edge in self.edge_switches}

            server_counter = 1
            # Sort edge switches for deterministic server naming
            for edge_switch_name in sorted(self.edge_switches):
                for _ in range(self.servers_per_edge):
                    server_name = f'server{server_counter}'
                    self.all_servers.append(server_name)
                    self.server_to_edge_map[server_name] = edge_switch_name
                    self.edge_to_servers_map[edge_switch_name].add(server_name)
                    server_counter += 1

            if not self.all_servers:
                raise ValueError("Network created with zero servers.")

            self.nodes = self.switches + self.all_servers

            self._create_all_potential_links()
            self._assign_capacities()
            self._build_adjacency_lists()

            self.network_json = self.generate_network_json(random_seed)
        else:
            raise ValueError("Either (k, servers_per_edge) or network_json must be provided.")

    def _create_all_potential_links(self):
        self.potential_links: Set[Tuple[str, str]] = set()

        for server, edge_switch in self.server_to_edge_map.items():
            self.potential_links.add((server, edge_switch))
            self.potential_links.add((edge_switch, server))

        for p_idx in range(1, self.num_pods + 1):
            for edge_sw in self.pod_to_edge_switches[p_idx]:
                for agg_sw in self.pod_to_agg_switches[p_idx]:
                    self.potential_links.add((edge_sw, agg_sw))
                    self.potential_links.add((agg_sw, edge_sw))
        
        core_sw_list = self.core_switches
        agg_switches_per_pod_count = self.k // 2 # Same as self.num_agg_switches_per_pod

        for p_idx in range(1, self.num_pods + 1):
            sorted_pod_agg_switches = sorted(list(self.pod_to_agg_switches[p_idx]))  # Sort to ensure deterministic order
            for agg_sw_local_idx, agg_sw_name in enumerate(sorted_pod_agg_switches):
                # target_core_block_start_idx defines the starting index (0-based) in core_sw_list for the block of core switches
                # this aggregation switch (agg_sw_name) will connect to.
                target_core_block_start_idx = agg_sw_local_idx * agg_switches_per_pod_count
                
                # Each aggregation switch connects to agg_switches_per_pod_count (k/2) core switches.
                # We iterate using a 0-based offset within the designated block of core switches.
                for core_block_offset in range(agg_switches_per_pod_count): # Iterates 0, 1, ..., (k/2 - 1)
                    # Calculate the 0-based index for core_sw_list
                    core_sw_list_idx = target_core_block_start_idx + core_block_offset
                    
                    # Ensure the calculated index is within the bounds of core_sw_list
                    if core_sw_list_idx < len(core_sw_list):
                        target_core_sw = core_sw_list[core_sw_list_idx]
                        self.potential_links.add((agg_sw_name, target_core_sw))
                        self.potential_links.add((target_core_sw, agg_sw_name))
        
        self.potential_links = sorted(self.potential_links)

    def _assign_capacities(self):
        """Assigns random bandwidths and processing capacities."""
        server_link_bandwidth = 200
        self.link_bandwidths: Dict[Tuple[str, str], float] = {}
        for u, v in self.potential_links:
            if u in self.all_servers or v in self.all_servers:
                self.link_bandwidths[(u,v)] = server_link_bandwidth
            else:
                #self.link_bandwidths[(u,v)] = self.rng.randint(200, 600)
                self.link_bandwidths[(u,v)] = 200

        self.switch_processing_capacities: Dict[str, float] = {}
        for switch in self.switches:
            #self.switch_processing_capacities[switch] = self.rng.randint(200, 600)
            self.switch_processing_capacities[switch] = 200

    def _build_adjacency_lists(self):
        self.outgoing_potential_links_map: Dict[str, Set[Tuple[str,str]]] = {node: set() for node in self.nodes}
        self.incoming_potential_links_map: Dict[str, Set[Tuple[str,str]]] = {node: set() for node in self.nodes}
        for u, v in self.potential_links:
            self.outgoing_potential_links_map[u].add((u, v))
            self.incoming_potential_links_map[v].add((u, v))

    def get_job_specific_links(self, job_info: Dict) -> Set[Tuple[str, str]]:
        """
        Returns the set of links relevant for shortest paths for a specific job in a Fat-Tree topology.
        This method iterates through all potential links and includes those that form part of a
        shortest path from a worker to the Parameter Server (PS), similar to the spine-leaf implementation.
        """
        ps_server = job_info['ps']
        ps_edge = job_info['ps_edge']
        workers = job_info['workers']
        job_links: Set[Tuple[str, str]] = set()

        # --- Pre-computation for efficient lookups ---
        if not hasattr(self, 'switch_to_pod_map'):
            self.switch_to_pod_map: Dict[str, int] = {}
            for pod, switches in self.pod_to_edge_switches.items():
                for s in switches: self.switch_to_pod_map[s] = pod
            for pod, switches in self.pod_to_agg_switches.items():
                for s in switches: self.switch_to_pod_map[s] = pod

        worker_edges = {self.server_to_edge_map[w] for w in workers}
        ps_pod = self.switch_to_pod_map.get(ps_edge)
        worker_pods = {self.switch_to_pod_map.get(e) for e in worker_edges if e in self.switch_to_pod_map}
        
        # Determine if traffic needs to cross pods (i.e., go up to core switches)
        inter_pod_traffic = any(wp != ps_pod for wp in worker_pods) or ps_pod is None

        # --- Iterate over all potential links to build the job-specific graph ---
        for u, v in self.potential_links:
            # 1. Worker -> Edge link (Uplink)
            if u in workers and v == self.server_to_edge_map.get(u):
                job_links.add((u, v))
                continue

            # 2. Edge -> Aggregation link (Uplink)
            if u in worker_edges and v in self.agg_switches:
                if self.switch_to_pod_map.get(u) == self.switch_to_pod_map.get(v):
                    job_links.add((u, v))
                    continue
            
            # 3. Aggregation -> Core link (Uplink)
            if inter_pod_traffic and u in self.agg_switches and v in self.core_switches:
                if self.switch_to_pod_map.get(u) in worker_pods:
                    job_links.add((u, v))
                    continue
            
            # --- Downlinks toward PS ---
            
            # 4. Core -> Aggregation link (Downlink)
            if inter_pod_traffic and u in self.core_switches and v in self.agg_switches:
                if self.switch_to_pod_map.get(v) == ps_pod:
                    job_links.add((u, v))
                    continue
            
            # 5. Aggregation -> Edge link (Downlink)
            if u in self.agg_switches and v == ps_edge:
                if self.switch_to_pod_map.get(u) == ps_pod:
                    job_links.add((u, v))
                    continue
            
            # 6. Edge -> PS Server link (Final Downlink)
            if u == ps_edge and v == ps_server:
                job_links.add((u, v))
                continue
                
        return job_links

    def get_links_from(self, node: str) -> Set[Tuple[str,str]]:
        return self.outgoing_potential_links_map.get(node, set())

    def get_links_to(self, node: str) -> Set[Tuple[str,str]]:
        return self.incoming_potential_links_map.get(node, set())

    def generate_network_json(self, random_seed: Optional[int] = None):
        network_data = {
            "metadata": {
                "type": "FatTreeNetwork",
                "k": self.k,
                "servers_per_edge": self.servers_per_edge,
                "random_seed": random_seed
            },
            "nodes": {
                "core_switches": sorted(list(self.core_switches)),
                "agg_switches": sorted(list(self.agg_switches)),
                "edge_switches": sorted(list(self.edge_switches)),
                "all_servers": sorted(list(self.all_servers)),
            },
            "edge_to_server_map": {edge: sorted(list(servers)) for edge, servers in sorted(self.edge_to_servers_map.items())},
            "links": {
                "potential_links_with_bandwidth": [
                    {"source": u, "target": v, "bandwidth": self.link_bandwidths.get((u,v))}
                    for u,v in sorted(self.potential_links)
                ],
            },
            "switch_processing_capacities": {
                s: c for s, c in sorted(self.switch_processing_capacities.items())
            }
        }
        return network_data
    
    def read_fat_tree_json_file(self, network_json: Dict):
        """
        Initializes the network state from a network JSON dictionary.
        """
        # Metadata
        meta = network_json['metadata']
        self.k = meta['k']
        self.servers_per_edge = meta['servers_per_edge']
        self.random_seed = meta.get('random_seed')
        self.rng = random.Random(self.random_seed)

        # Nodes
        nodes_data = network_json['nodes']
        self.core_switches = sorted(nodes_data['core_switches'])
        self.agg_switches = sorted(nodes_data['agg_switches'])
        self.edge_switches = sorted(nodes_data['edge_switches'])
        self.switches = sorted(self.core_switches + self.agg_switches + self.edge_switches)
        self.all_servers = sorted(nodes_data['all_servers'])
        self.nodes = self.switches + self.all_servers

        # Reconstruct mappings
        self.edge_to_servers_map = {
            edge: set(servers) for edge, servers in network_json['edge_to_server_map'].items()
        }
        self.server_to_edge_map = {}
        for edge, servers in self.edge_to_servers_map.items():
            for server in servers:
                self.server_to_edge_map[server] = edge

        # Pod-related attributes
        self.num_pods = self.k
        self.pod_to_agg_switches: Dict[int, Set[str]] = {p: set() for p in range(1, self.num_pods + 1)}
        self.pod_to_edge_switches: Dict[int, Set[str]] = {p: set() for p in range(1, self.num_pods + 1)}
        for switch in self.agg_switches:
            pod_num = int(switch.split('_')[0].replace('pod', ''))
            self.pod_to_agg_switches[pod_num].add(switch)
        for switch in self.edge_switches:
            pod_num = int(switch.split('_')[0].replace('pod', ''))
            self.pod_to_edge_switches[pod_num].add(switch)
            
        # Links and Bandwidths
        self.potential_links = []
        self.link_bandwidths = {}
        if 'links' in network_json and 'potential_links_with_bandwidth' in network_json['links']:
            for link_data in network_json['links']['potential_links_with_bandwidth']:
                u, v = link_data['source'], link_data['target']
                link_tuple = (u, v)
                self.potential_links.append(link_tuple)
                if 'bandwidth' in link_data and link_data['bandwidth'] is not None:
                    self.link_bandwidths[link_tuple] = link_data['bandwidth']
        self.potential_links = sorted(self.potential_links)

        # Capacities
        self.switch_processing_capacities = network_json['switch_processing_capacities']

        # Adjacency lists
        self._build_adjacency_lists()

        # Store the original JSON
        self.network_json = network_json

# --- Job Definition Function (Adapted for FatTreeNetwork) ---
def fat_tree_jobs_placement(network: FatTreeNetwork,
                            job_specs: List[Dict],
                            random_seed: Optional[int] = None) -> List[Dict]:
    rng = random.Random(random_seed)
    available_servers = network.all_servers
    assigned_servers = set()
    jobs_defined = []
    job_id_counter = 1

    total_servers_needed = sum(spec['num_workers'] + 1 for spec in job_specs)
    if total_servers_needed > len(available_servers):
        raise ValueError(f"Not enough servers ({len(available_servers)}) for jobs (need {total_servers_needed}).")

    assigned_ps_servers = set()
    for spec in job_specs:
        num_workers = spec['num_workers']
        servers_needed = num_workers + 1
        
        temp_available = [s for s in available_servers if s not in assigned_servers]
        if len(temp_available) < servers_needed:
             raise ValueError(f"Cannot allocate {servers_needed} unique servers for job {job_id_counter}.")

        job_servers = rng.sample(temp_available, servers_needed)
        
        ps_candidate = None
        shuffled_job_servers = list(job_servers)
        rng.shuffle(shuffled_job_servers)
        for server in shuffled_job_servers:
            if server not in assigned_ps_servers:
                ps_candidate = server
                break
        if ps_candidate is None:
             raise ValueError(f"Could not assign unique PS for job {job_id_counter}.")
        
        ps_server = ps_candidate
        assigned_ps_servers.add(ps_server)
        worker_servers = set(job_servers) - {ps_server}
        assigned_servers.update(job_servers)

        job_info = {
            'id': job_id_counter,
            'workers': worker_servers,
            'ps': ps_server,
            'ps_edge': network.server_to_edge_map[ps_server] # Changed from ps_leaf and server_to_leaf_map
        }
        jobs_defined.append(job_info)
        job_id_counter += 1
    
    return jobs_defined

def generate_jobs_json(jobs):
    jobs_json = []
    for job in jobs:
        job_json = {
            "id": job['id'],
            "workers": sorted(list(job['workers'])),
            "ps": job['ps'],
            "ps_edge": job['ps_edge']
        }
        jobs_json.append(job_json)
    return jobs_json

def generate_fat_tree_jobs_json(output_dir: str, network: FatTreeNetwork, jobs: List[Dict], my_seed: int) -> Dict:
    network_json = network.network_json
    jobs_json = generate_jobs_json(jobs)
    # --- Save Network and Jobs to JSON File ---
    data_to_save = {
        "network_json": network_json,
        "jobs_json": jobs_json
    }

    k = network.k
    edge_num = len(network.edge_switches)
    server_num = network.servers_per_edge * edge_num

    job_num = len(jobs)

    output_filename = f"fat_tree_pod{k}_servers{server_num}_jobs{job_num}_seed{my_seed}.json"

    output_filepath = os.path.join(output_dir, output_filename)

    try:
        with open(output_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4, sort_keys=False) # sort_keys for consistent file
        print(f"\nSuccessfully wrote combined topology and jobs to: {output_filepath}")
    except IOError as e:
        print(f"\nError writing to file {output_filepath}: {e}")
    except TypeError as e:
        print(f"\nError serializing data to JSON: {e}. Check data structures in network_json or jobs_json_data.")

    return {
        "network_json": network_json,
        "jobs_json": jobs_json
    }