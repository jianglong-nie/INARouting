import os
import json
import random
from typing import Optional, List, Dict, Set, Tuple

# --- Network Topology Class (Refined Link Handling) ---
class SpineLeafNetwork:
    """
    Represents the Spine-Leaf network topology with Servers.
    Stores nodes, directed links, link bandwidths, and switch processing capacities.
    Does NOT define job-specific workers/PS. This is handled separately.
    """
    def __init__(self,
                 leaf_switch_num: Optional[int] = None,
                 spine_switch_num: Optional[int] = None,
                 servers_per_leaf: Optional[int] = None,
                 random_seed: Optional[int] = None,
                 network_json: Optional[Dict] = None):
        if network_json is not None:
            self.read_spine_leaf_json_file(network_json)
        elif leaf_switch_num is not None and spine_switch_num is not None and servers_per_leaf is not None:
            if servers_per_leaf <= 0:
                raise ValueError("servers_per_leaf must be positive.")
            if leaf_switch_num <= 0 or spine_switch_num <= 0:
                raise ValueError("Switch numbers must be positive.")

            self.leaf_switch_num = leaf_switch_num
            self.spine_switch_num = spine_switch_num
            self.servers_per_leaf = servers_per_leaf
            self.rng = random.Random(random_seed)

            # Node Creation
            self.leaf_switches = [f'leaf{i}' for i in range(1, leaf_switch_num + 1)]
            self.spine_switches = [f'spine{i}' for i in range(1, spine_switch_num + 1)]
            self.switches = self.leaf_switches + self.spine_switches
            self.switches = sorted(self.switches)

            self.all_servers = set()
            self.server_to_leaf_map = {}
            self.leaf_to_servers_map = {leaf: set() for leaf in self.leaf_switches}

            server_counter = 1
            for leaf in self.leaf_switches:
                for _ in range(servers_per_leaf):
                    server_name = f'server{server_counter}'
                    self.all_servers.add(server_name)
                    self.server_to_leaf_map[server_name] = leaf
                    self.leaf_to_servers_map[leaf].add(server_name)
                    server_counter += 1

            if not self.all_servers:
                raise ValueError("Network created with zero servers.")

            self.nodes = sorted(list(self.all_servers)) + self.switches
            self.server_num = len(self.all_servers)

            # Create ALL potential directed links upfront
            self._create_all_potential_links()

            # Assign Weights/Capacities
            self._assign_capacities()

            # Precompute adjacency lists (based on ALL potential links)
            self._build_adjacency_lists()

            self.network_json = self.generate_network_json(random_seed)
        else:
            raise ValueError("Either network_json or (leaf_switch_num, spine_switch_num, servers_per_leaf) must be provided.")

    def _create_all_potential_links(self):
        """ Creates a set of ALL potentially usable directed links."""
        self.potential_links = set()
        # Server <-> Leaf links
        for server, leaf in self.server_to_leaf_map.items():
            self.potential_links.add((server, leaf))
            self.potential_links.add((leaf, server))
        # Leaf <-> Spine links
        for leaf in self.leaf_switches:
            for spine in self.spine_switches:
                self.potential_links.add((leaf, spine))
                self.potential_links.add((spine, leaf))
        
        self.potential_links = sorted(self.potential_links)


    def _assign_capacities(self):
        """Assigns random bandwidths and processing capacities."""
        server_link_bandwidth = 200
        self.link_bandwidths = {}
        for link in self.potential_links:
            u, v = link
            if u in self.all_servers or v in self.all_servers:
                self.link_bandwidths[link] = server_link_bandwidth
            else:
                #self.link_bandwidths[link] = self.rng.randint(200, 600)
                self.link_bandwidths[link] = 200

        self.switch_processing_capacities = {}
        for switch in self.switches:
            #self.switch_processing_capacities[switch] = self.rng.randint(200, 600)
            self.switch_processing_capacities[switch] = 200

    def _build_adjacency_lists(self):
        """Helper method based on all potential_links."""
        self.outgoing_potential_links_map = {node: set() for node in self.nodes}
        self.incoming_potential_links_map = {node: set() for node in self.nodes}
        for u, v in self.potential_links:
            self.outgoing_potential_links_map[u].add((u, v))
            self.incoming_potential_links_map[v].add((u, v))

    def get_job_specific_links(self, job_info: Dict) -> Set[Tuple[str, str]]:
        """
        Returns the set of links relevant for shortest paths for a specific job,
        filtering from all potential links.
        """
        ps_leaf = job_info['ps_leaf']
        ps_server = job_info['ps']
        workers = job_info['workers']
        job_links = set()

        # Iterate through ALL potential links and keep only those matching the shortest path structure
        for u, v in self.potential_links:
            # 1. Worker -> Leaf link?
            if u in workers and v == self.server_to_leaf_map.get(u):
                job_links.add((u, v))
            # 2. Leaf -> Spine link? (Only if Leaf is NOT ps_leaf and has workers for THIS job)
            elif u in self.leaf_switches and u != ps_leaf and v in self.spine_switches:
                 # Check if this leaf 'u' is connected to any worker of *this* job
                 if any(w in self.leaf_to_servers_map.get(u, set()) for w in workers):
                     job_links.add((u, v))
            # 3. Spine -> Leaf link? (Only if Leaf IS ps_leaf)
            elif u in self.spine_switches and v == ps_leaf:
                 job_links.add((u, v))
            # 4. PS_Leaf -> PS Server link?
            elif u == ps_leaf and v == ps_server:
                 job_links.add((u, v))

        return job_links

    # --- get_links_from / get_links_to now operate purely on potential links ---
    def get_links_from(self, node: str) -> set:
        """Returns potential outgoing links (tuples) from the given node."""
        return self.outgoing_potential_links_map.get(node, set())

    def get_links_to(self, node: str) -> set:
        """Returns potential incoming links (tuples) to the given node."""
        return self.incoming_potential_links_map.get(node, set())

    def generate_network_json(self, random_seed: Optional[int] = None):
        network_data = {
            "metadata": {
                "type": "SpineLeafNetwork",
                "leaf_switch_num": self.leaf_switch_num,
                "spine_switch_num": self.spine_switch_num,
                "servers_per_leaf": self.servers_per_leaf,
                "random_seed": random_seed
            },
            "nodes": {
                "leaf_switches": sorted(list(self.leaf_switches)), # Sort for consistent output
                "spine_switches": sorted(list(self.spine_switches)),
                "all_servers": sorted(list(self.all_servers)),
            },
            "server_to_leaf_map": {s: l for s, l in sorted(self.server_to_leaf_map.items())},
            "leaf_to_servers_map": {
                leaf: sorted(list(servers)) for leaf, servers in sorted(self.leaf_to_servers_map.items())
            },
            "links": [
                {"source": u, "target": v, "bandwidth": self.link_bandwidths.get((u,v))}
                for u,v in sorted(self.potential_links)
            ],
            "switch_processing_capacities": {
                s: c for s, c in sorted(self.switch_processing_capacities.items())
            }
        }
        return network_data

    def read_spine_leaf_json_file(self, network_json):
        """
        Initializes the network state from a network JSON dictionary.
        This method re-initializes the object.
        """
        # Metadata
        meta = network_json['metadata']
        self.leaf_switch_num = meta['leaf_switch_num']
        self.spine_switch_num = meta['spine_switch_num']
        self.servers_per_leaf = meta['servers_per_leaf']
        self.random_seed = meta.get('random_seed')
        self.rng = random.Random(self.random_seed)

        # Nodes
        nodes_data = network_json['nodes']
        self.leaf_switches = sorted(nodes_data['leaf_switches'])
        self.spine_switches = sorted(nodes_data['spine_switches'])
        self.switches = sorted(self.leaf_switches + self.spine_switches)
        self.all_servers = sorted(nodes_data['all_servers'])
        self.server_num = len(self.all_servers)
        self.nodes = self.all_servers + self.switches

        # Mappings
        self.server_to_leaf_map = network_json['server_to_leaf_map']
        self.leaf_to_servers_map = {
            leaf: set(servers)
            for leaf, servers in network_json['leaf_to_servers_map'].items()
        }

        # Links and Bandwidths
        self.potential_links = []
        self.link_bandwidths = {}
        if network_json.get('links'):
            for link_data in network_json['links']:
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

# --- Job Definition Function (Unchanged) ---
def spine_leaf_jobs_placement(network: SpineLeafNetwork,
                              job_specs: List[Dict],
                              random_seed: Optional[int] = None) -> List[Dict]:
    """
    Defines job details (workers, PS) based on specifications and network topology.
    Ensures deterministic placement if a random_seed is provided.

    Args:
        network: The SpineLeafNetwork instance.
        job_specs: A list of dictionaries, e.g., [{'num_workers': 3}, {'num_workers': 7}]
        random_seed: An optional integer to seed the random number generator for
                     deterministic server selection. If None, selection is random.

    Returns:
        A list of job information dictionaries, including assigned workers and PS.
        Returns empty list or raises error if not enough servers.
    """
    # Create a local random number generator instance
    rng = random.Random(random_seed)

    available_servers = list(network.all_servers)
    # Sort the list to ensure deterministic order before sampling
    available_servers.sort()

    # rng.shuffle(available_servers) # Use rng if shuffling the initial list is desired
    assigned_servers = set()
    jobs_defined = []
    job_id_counter = 1

    total_servers_needed = sum(spec['num_workers'] + 1 for spec in job_specs)
    if total_servers_needed > len(available_servers):
        raise ValueError(f"Not enough servers ({len(available_servers)}) available for requested jobs (need {total_servers_needed}).")

    assigned_ps_servers = set()

    for spec in job_specs:
        num_workers = spec['num_workers']
        servers_needed = num_workers + 1
        job_servers = []

        # Find unique servers for this job
        temp_available = [s for s in available_servers if s not in assigned_servers]
        if len(temp_available) < servers_needed:
             raise ValueError(f"Cannot allocate {servers_needed} unique servers for job {job_id_counter}.") # Should be caught earlier

        # Use the local rng for sampling
        job_servers = rng.sample(temp_available, servers_needed)

        # Select PS ensuring it's unique across jobs
        ps_candidate = None
        shuffled_job_servers = list(job_servers) # Create a mutable copy
        # Use the local rng for shuffling
        rng.shuffle(shuffled_job_servers)
        for server in shuffled_job_servers:
            if server not in assigned_ps_servers:
                ps_candidate = server
                break
        if ps_candidate is None:
             raise ValueError(f"Could not assign a unique PS server for job {job_id_counter} from the selected servers {job_servers}. This might indicate an issue if all selected servers were already PS for other jobs.") # Should be rare

        ps_server = ps_candidate
        assigned_ps_servers.add(ps_server)
        worker_servers = set(job_servers) - {ps_server}

        # Update assigned servers for the next job
        assigned_servers.update(job_servers)

        job_info = {
            'id': job_id_counter,
            'workers': worker_servers,
            'ps': ps_server,
            'ps_leaf': network.server_to_leaf_map[ps_server]
        }
        jobs_defined.append(job_info)
        job_id_counter += 1

    # network.jobs = jobs_defined # Remove the hacky assignment
    return jobs_defined

def generate_jobs_json(jobs):
    jobs_json = []
    for job in jobs:
        job_json = {
            "id": job['id'],
            "workers": sorted(list(job['workers'])),
            "ps": job['ps'],
            "ps_leaf": job['ps_leaf']
        }
        jobs_json.append(job_json)
    return jobs_json

def generate_spine_leaf_jobs_json(output_dir: str, network: SpineLeafNetwork, jobs: List[Dict], my_seed: int) -> Dict:
    network_json = network.network_json
    jobs_json = generate_jobs_json(jobs)
    # --- Save Network and Jobs to JSON File ---
    data_to_save = {
        "network_json": network_json,
        "jobs_json": jobs_json
    }
    spine_switch_num = network.spine_switch_num
    leaf_switch_num = network.leaf_switch_num

    server_num = network.servers_per_leaf * leaf_switch_num

    job_num = len(jobs)

    output_filename = f"spine{spine_switch_num}_leaf{leaf_switch_num}_servers{server_num}_jobs{job_num}_seed{my_seed}.json"

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