# Some standard imports
import numpy as np
from torch import nn
import torch.nn.init as init
import torch.onnx
import onnx
from onnx import shape_inference
from google.protobuf.json_format import MessageToDict
from utils import *
from estimate import *
from onnx_parser import *
from torchprofile import profile_macs

if __name__ == '__main__':
    print("*** Load the NN model ***")

    # Load the NN model
    model = onnx.load("super_resolution.onnx")

    # Run the shape inference to get the node dimensions (e.g. channels)
    inferred_model = compute_shape_inference(model)

    # Extract the input dimensions
    input_arr = extract_input_dimensions(model)

    # Extract the shape_inferred dimensions
    shape_list = extract_shape_inferred_dimensions(inferred_model)

    # Extract the convolutional filters
    filter_list = extract_conv_filters(model)

    # Extract the operation type and other parameters
    layers = parse_nn_model(model)

    # Append the dimension information for each layer in the network
    layers = append_dimensions(input_arr, shape_list, layers)

    # Append the number of filters for each convolutional layer in the network
    layers = append_filters(filter_list, layers)

    for index in range(len(layers)):
        # Print to visualize the results
        print(layers[index])

    print("/////////////////////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////////////////////")

    # Get the processor configuration
    processor_config = User_Input()
    processor_config.set_parameters(pipeline_en = "Enabled",
                               pipeline_stages = 5,
                               lu_en = "Enabled",
                               lu_factor = 3,
                               data_forwarding = "Enabled",
                               memory_latency = 3,
                               addition = 2,
                               multiplication = 1,
                               division = 1,
                               nr_adders = 0,
                               nr_multipliers = 0,
                               SIMD_status = "Disabled",
                               VLIW_status = "Disabled")

    # Initialize the network clock cycles  
    network_cc = 0

    # # Calculate the number of clock cycles for each layer
    # for index in range(len(layers)):
    #     if layers[index].operation_type == 'Conv':
    #         layer_cc = calculate_conv_cc_2(layers[index], processor_config)
    #         print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
    #     elif layers[index].operation_type == 'ReLU':
    #         layer_cc = calculate_relu_cc(layers[index], processor_config, 5)
    #         print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
    #     elif layers[index].operation_type == 'MaxPool':
    #         layer_cc = calculate_max_pool_cc(layers[index], processor_config, 5)
    #         print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
    #     elif layers[index].operation_type == 'AvgPool':
    #         layer_cc = calculate_avg_pool_cc(layers[index], processor_config, 5)
    #         print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
    #     network_cc += layer_cc

    # layers[0].in_size = (2,2,1,0)
    # layers[0].out_channels = (1)
    # layer_cc = calculate_conv_cc(layers[0], processor_config, 2)
    # print(f"Clock cycles conv after: {layer_cc}")
    # layer_cc = calculate_conv_cc_2(layers[0], processor_config)
    # print(f"Clock cycles conv2 after: {layer_cc}")


    layers[0].in_size = 32
    layers[0].out_size = 2

    layer_cc = calculate_fully_connected_cc(layers[0], processor_config, 5)
    print(f"Clock cycles conv2 after: {layer_cc}")