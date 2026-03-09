import json
import ast
import os

with open('/home/francesco/Universita/PhD/PROJECTS/CancerEvo/simulation_chrom_change.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

utils_cells = []
interventions_cells = []
simulation_cells = []
parameters_cells = []

for i, cell in enumerate(nb.get('cells', [])):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if not source.strip():
            continue
            
        if 'import Pkg' in source or 'using Random' in source:
            utils_cells.append(source)
            simulation_cells.append(source)
            interventions_cells.append(source)
            
        elif 'function save_to_file' in source:
            utils_cells.append(source)
        elif 'mutable struct Chromosomes' in source:
            utils_cells.append(source)
        elif 'mutable struct Cell' in source:
            utils_cells.append(source)
        elif 'mutable struct Tissue' in source:
            utils_cells.append(source)
        elif 'function substitute!' in source:
            utils_cells.append(source)
        elif 'function get_r(' in source:
            utils_cells.append(source)
        elif 'function create_gene_map(' in source:
            utils_cells.append(source)
            
        elif 'function int_a!' in source:
            interventions_cells.append(source)
            
        # exclude plot functions
        elif 'plot_' in source or 'heatmap' in source or 'Plots.' in source:
            continue
            
        elif 'function simulation(' in source:
            # We also need Results struct, which is in the same cell in the notebook
            simulation_cells.append(source)
            
        elif 'N_I  = 10' in source and 'N_O  = 10' in source:
            parameters_cells.append(source)

os.makedirs('/home/francesco/Universita/PhD/PROJECTS/CancerEvo/scripts', exist_ok=True)
os.makedirs('/home/francesco/Universita/PhD/PROJECTS/CancerEvo/data/simulations', exist_ok=True)

with open('/home/francesco/Universita/PhD/PROJECTS/CancerEvo/scripts/utils.jl', 'w') as f:
    f.write("\n\n".join(utils_cells))
    
with open('/home/francesco/Universita/PhD/PROJECTS/CancerEvo/scripts/interventions.jl', 'w') as f:
    f.write('include("utils.jl")\n\n')
    f.write("\n\n".join(interventions_cells))
    
with open('/home/francesco/Universita/PhD/PROJECTS/CancerEvo/scripts/simulation.jl', 'w') as f:
    f.write('include("utils.jl")\n')
    f.write('include("parameters.jl")\n')
    f.write('include("interventions.jl")\n\n')
    f.write("\n\n".join(simulation_cells))
    
with open('/home/francesco/Universita/PhD/PROJECTS/CancerEvo/scripts/parameters.jl', 'w') as f:
    f.write("\n\n".join(parameters_cells))

print("Extraction complete.")
