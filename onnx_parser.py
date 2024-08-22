# Some standard imports
import numpy as np
from onnx import shape_inference
from google.protobuf.json_format import MessageToDict
from utils import *


def extract_input_dimensions(model):
    print("*** Extract the input size of the NN model ***") 

    input_size_list = []

    # Extract the input dimensions
    for _input in model.graph.input:
        input_dim = _input.type.tensor_type.shape.dim
        for dimension in input_dim:
            input_size_list.append(dimension.dim_value)

        input_arr = np.flip(np.array(input_size_list))
        input_size_list.clear()

    return input_arr

def compute_shape_inference(model):
    print("*** Compute the shape inferrence for the given model ***") 
    
    inferred_model = shape_inference.infer_shapes(model)

    return inferred_model

def extract_shape_inferred_dimensions(model):
    print("*** Extract the shape inferred sizes ***")

    # List with all inferred shapes
    shape_list      = []
    # List with all input sizes
    input_size_list = []
    
    for _layer in model.graph.value_info:
        # Create a shape_inferred object to store the informations
        shape_info = shape_inferred()
        # Extract the name of the layer
        shape_info.name = '[\'' + str(_layer.name) + '\']'
        # Extract the input dimension of the next layer
        shape = _layer.type.tensor_type.shape.dim
        for dimension in shape:
            input_size_list.append(dimension.dim_value)
        shape_info.dimensions = np.flip(np.array(input_size_list))
        input_size_list.clear()
        shape_list.append(shape_info)
    
    return shape_list

# Helper function to extract the number of filters for a CONV layer
def extract_conv_filters(model):
    print("*** Extract the number of filters for CONV layers ***")
    # List with all convolutional filters
    filters_list = []

    # Iterate and extract the layers parameters:
    for _input in model.graph.initializer:
        if "conv" in str({_input.name}) and  "bias" in str({_input.name}) :
            text_name = _input.name
            element = filters_struct()
            element.set_parameters(text_name.split(".", 1)[0], _input.dims)
            filters_list.append(element)
    return filters_list

# Main method to extract all information required from the ONNX model
def parse_nn_model(model):
    print("*** Extract the layers informations ***")

    # List with all layers information
    layers = []

    # Iterate and extract the layers parameters:
    for _input in model.graph.node:
        # Create a new layer object
        layer_info = layer_information()

        # Convolutional layer
        if _input.op_type == "Conv":
            layer_info.operation_type = "Conv"
            layer_info.name = str(_input.output)

            # Extract the conv2D parameters
            attr_list = _input.attribute
            for attribute in attr_list:
                if attribute.name == "dilations":
                    layer_info.dilation = attribute.ints
                elif attribute.name == "kernel_shape":
                    layer_info.kernel_size = attribute.ints
                elif attribute.name == "pads":
                    layer_info.padding = attribute.ints
                elif attribute.name == "strides":
                    layer_info.stride = attribute.ints
                
            # Put the element in the list
            layers.append(layer_info)

        # MaxPool layer
        elif  _input.op_type == "MaxPool":
            layer_info.operation_type = "MaxPool"
            layer_info.name = str(_input.output)

            # Extract the MaxPool2D parameters
            attr_list = _input.attribute
            for attribute in attr_list:
                if attribute.name == "kernel_shape":
                    layer_info.kernel_size = attribute.ints
                elif attribute.name == "pads":
                    layer_info.padding = attribute.ints
                elif attribute.name == "strides":
                    layer_info.stride = attribute.ints

            # Put the element in the list
            layers.append(layer_info)

        # AveragePool layer
        elif  _input.op_type == "AveragePool":
            layer_info.operation_type = "AvgPool"
            layer_info.name = str(_input.output)

            # Extract the AvgPool2D parameters
            attr_list = _input.attribute
            for attribute in attr_list:
                if attribute.name == "kernel_shape":
                    layer_info.kernel_size = attribute.ints
                elif attribute.name == "pads":
                    layer_info.padding = attribute.ints
                elif attribute.name == "strides":
                    layer_info.stride = attribute.ints

            # Put the element in the list
            layers.append(layer_info)

        # Linear - fully connected layer
        elif  _input.op_type == "Gemm":
            layer_info.operation_type = "Gemm"
            layer_info.name = str(_input.output)

            # Put the element in the list
            layers.append(layer_info)

        # ReLU - activation function
        elif  _input.op_type == "Relu":
            layer_info.operation_type = "ReLU"
            layer_info.name = str(_input.output)

            # Put the element in the list
            layers.append(layer_info)

        # Flatten
        elif  _input.op_type == "Flatten":
            layer_info.operation_type = "Flatten"
            layer_info.name = str(_input.output)

            # Put the element in the list
            layers.append(layer_info)

        # TODO: Discuss about these type of special operation
        # BatchNorm
        elif  _input.op_type == "BatchNormalization":
            layer_info.operation_type = "BatchNorm"
            layer_info.name = str(_input.output)

            # Put the element in the list
            layers.append(layer_info)

        # # Dropout
        # elif  _input.op_type == "Dropout":
        #     layer_info.operation_type = "Dropout"
        #     layer_info.name = str(_input.output)

        #     # Put the element in the list
        #     layers.append(layer_info)

    return layers

# Helper function to append the correct input size for each layer
def append_dimensions(input_arr, shape_list, layers):
    # Take the dimension of the first layer from the input
    layers[0].in_size = input_arr

    for index in range(1, len(layers)):
        # Iterate through the shape_list variable and extract the correct dimensions
        for index2 in range(len(shape_list)):
            if layers[index].name == shape_list[index2].name:
                # The input dimension of the current layer is equal to the output dimension of the previous layer
                layers[index].in_size = shape_list[index2 - 1].dimensions

    return layers

# Helper function to append the information regarding the filters in conv layer
def append_filters(filter_list, layers):
    for index in range(len(layers)):
        # Iterate through the filter_list variable and extract the correct number of filters
        for index2 in range(len(filter_list)):
            if layers[index].name.split("/", 2)[1] == filter_list[index2].name:
                # The out_channels for the current CONV layer is equal to the value of the filter_list element
                layers[index].out_channels = filter_list[index2].value

    return layers
