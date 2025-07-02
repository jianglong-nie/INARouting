# INARouting: Efficient Multi-Job Routing Optimization for Hierarchical In-Network Aggregation

A simulation and evaluation framework for INARouting algorithms in spine-leaf and fat-tree network. This project provides tools to generate network topologies, run various ours algorithms, and evaluate their performance.

## Background

In-network aggregation (INA) has emerged as a key technology to alleviate communication bottlenecks in large-scale distributed training, 
but its performance is often hindered by suboptimal routing. 
Existing INA-aware routing algorithms suffer from certain limitations: 
they either lack a global, multi-job coordination mechanism, 
or operate on incomplete network models that ignore key hardware constraints such as switch processing capacity. 
These deficiencies lead to network congestion and inefficient resource utilization, 
ultimately undermining the full potential of INA.
To address these challenges, 
we present **INARouting**, a novel framework that holistically solves the multi-job hierarchical aggregation routing problem. 
We propose a comprehensive Mixed-Integer Linear Program (MILP) that, for the first time, 
co-optimizes routing for all concurrent jobs under a unified fair-sharing principle for both link bandwidth and switch processing capacity.
To address different deployment scenarios, we develop two variants: 
**INARouting-Opt** that provides optimal solutions for moderate-scale networks, 
and **INARouting-Relax**, a fast and effective heuristic using LP-relaxation and a greedy score-based rounding algorithm for large-scale deployments.


## Project Structure

Here is an overview of the project's directory structure:

```
INARouting/
├── algorithms/         # implement INARouting algorithms (INARouting-Opt:ours_opt.py & INARouting-Relax:ours.py)
├── network_generator/  # Scripts to generate network topologies like Fat-Tree and Spine-Leaf.
├── evaluation/         # Scripts to parse results, calculate metrics (throughput, JCT), and generate plots.
├── generated_data/     # Default directory for storing generated network topology files.
├── routing_results/    # Default directory for storing raw simulation outputs from all algorithms.
├── ours_results/       # Directory for storing results specifically from the INARouting algorithm.
└── README.md           # This document.
```

## Prerequisites

Before you begin, ensure you have the following installed:
*   Python (e.g., 3.8 or newer)
*   Gurobi 

All other Python dependencies are listed in `requirements.txt`.

## Installation

Follow these steps to set up the project environment:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jianglong-nie/INARouting
    cd INARouting
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
    > **Important Note:** This project requires the Gurobi optimizer. In addition to installing the gurobipy package via pip, you also need to:
    > 1. Visit https://www.gurobi.com/ (or Chinese website http://www.gurobi.cn/) to register an account
    > 2. Apply for and obtain a Gurobi license
    > 3. Configure the license according to the official documentation
    > 
    > Academic users can apply for a free academic license.

## How to Run

We provide two approaches to experience and validate the effectiveness of the INARouting algorithm:

## Option 1: View Pre-computed Results (Recommended for Quick Experience)

We have pre-run the INARouting algorithm and saved the routing results for your convenience. You can directly run the evaluation scripts to view the algorithm's performance:

**For Spine-Leaf network topology:**
```bash
python evaluation/run_spine_leaf.py
```

**For Fat-Tree network topology:**
```bash
python evaluation/run_fat_tree.py
```

## Option 2: Run the Full Pipeline from Scratch

This option allows you to re-generate all routing results by running the optimization algorithms from scratch.

### Step 1: Generate New Routing Results

Running the scripts below will execute the core optimization algorithms. The new routing solutions will be saved in `INARouting/routing_results/`, overwriting any previous results. The scripts likely require arguments to specify the network topology.

**To generate results for Spine-Leaf topology:**
```bash
# Run the optimal algorithm (INARouting-Opt), which is much slower
python algorithms/get_spine_leaf_ours_opt.py

# Run the heuristic algorithm (INARouting-Relax)
python algorithms/get_spine_leaf_ours.py
```

**To generate results for Fat-Tree topology:**
```bash
# Run the optimal algorithm (INARouting-Opt), which is much slower
python algorithms/get_fat_tree_ours_opt.py

# Run the heuristic algorithm (INARouting-Relax)
python algorithms/get_fat_tree_ours.py
```

### Step 2: Evaluate the New Results

After the routing results have been re-generated, run the same evaluation scripts from Option 1 to visualize the performance of the new solutions.

**For Spine-Leaf:**
```bash
python evaluation/run_spine_leaf.py
```

**For Fat-Tree:**
```bash
python evaluation/run_fat_tree.py
```