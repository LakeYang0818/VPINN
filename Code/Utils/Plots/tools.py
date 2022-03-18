from datetime import datetime
import os
import yaml

output_dir = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
os.makedirs(os.path.join(os.path.join(os.getcwd(), 'Results'), output_dir))

"""Plot utilities"""


# Generate a text box string from the config
def info_from_cfg(cfg):
    dim = cfg['space']['dimension']

    pde_info = rf"{dim}-dimensional {cfg['PDE']['type']} equation"
    net_info = f"Neural net: " + \
               f"{cfg['architecture']['layers']} layers, {cfg['architecture']['nodes_per_layer']} neurons per layer "

    grid_info = f"Grid: ${cfg['space']['boundary']['x']}$"
    if dim == 2:
        grid_info += fr" $\times {cfg['space']['boundary']['y']}$"
    grid_info += fr", $n_x=${cfg['space']['grid_size']['x']}"
    if dim == 2:
        grid_info += fr", $n_y=${cfg['space']['grid_size']['y']}"
    var_form_info = f"Variational form: {cfg['variational_form']}"
    test_func_info = fr"Number of test functions: $K_x=$ {cfg['Test functions']['N_test_functions']['x']}"
    if dim == 2:
        test_func_info += fr", $K_y=$ {cfg['Test functions']['N_test_functions']['y']}"
    test_func_info += '\n' + fr"Test function type: {cfg['Test functions']['type']}"
    iterations_info = fr"{cfg['N_iterations']} iterations, $\tau_b=${cfg['boundary_loss_weight']}" + \
                      fr", $\tau_v=${cfg['variational_loss_weight']}"

    return pde_info + '\n \n' + '\n'.join([net_info, grid_info, var_form_info, test_func_info, iterations_info])


# Write the config
def write_config(cfg):
    with open(f'Results/' + output_dir + '/config.yaml', 'w') as file:
        yaml.dump(cfg, file)
