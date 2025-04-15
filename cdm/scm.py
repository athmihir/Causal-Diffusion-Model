from torch import nn
import torch
import math
from enum import Enum
from collections import defaultdict
from constants import IMAGE_CHANNELS, IMAGE_RESOLUTION

class EdgeType(Enum):
    DIRECTED = 1
    BI_DIRECTED = 2

class Vertex():
    def __init__(self, label, dimensions):
        self.label = label
        self.dimensions = dimensions
        self.deriving_function = None

    def __hash__(self):
        return hash((self.label))
    
    def __eq__(self, other):
        return self.label == other.label and self.dimensions == other.dimensions and self.deriving_function == other.deriving_function

class Edge():
    def __init__(self, src_vertex:Vertex, dest_vertex:Vertex, edge_type=EdgeType):
        self.src_vertex = src_vertex
        self.dest_vertex = dest_vertex
        self.edge_type = edge_type

class SCM(nn.Module):
    def __init__(self, edges: list[Edge]):
        super().__init__()
        # Holds the main DAG
        self.functional_map = defaultdict(list)
        # Holds the computed value tensors for the forward passes.
        self.value_map = {}
        # construct the functional map 
        self.construct_functional_map(edges)
        # define the ncm functional components for each vertex in the functional map
        self.define_functional_components()
    
    def define_functional_components(self):
        '''Defines the NCM functional relationships between input and output vertex'''
        for vertex, input_list in self.functional_map.items():
            # ie. vertex does not represent exogenous U.
            if vertex.deriving_function is not None:
                # These are the total channels after combining all inputs to the node.
                total_channels = 0
                for element in input_list:
                    total_channels += element.dimensions[0]
                vertex.deriving_function = SCMFunction((total_channels, IMAGE_RESOLUTION, IMAGE_RESOLUTION), output_classes=vertex.dimensions)

    def construct_functional_map(self, edges: list[Edge]):
        '''Constructs the reverse adjacency list for the DAG.
        Reverse because for every node, we need the results from its ancestors during the forward computation.'''
        for edge in edges:
            if edge.edge_type is EdgeType.DIRECTED:
                # Add functional relationship from destination to source
                self.functional_map[edge.dest_vertex].append(edge.src_vertex)
            else:
                # Create a vertex for unobserved confounding and add it to both the edges
                uc_vertex = Vertex(f"U_{edge.dest_vertex.label}_{edge.src_vertex.label}", dimensions=(IMAGE_CHANNELS, IMAGE_RESOLUTION, IMAGE_RESOLUTION))
                uc_vertex.deriving_function = lambda x: torch.randn(uc_vertex.dimensions)
                self.functional_map[edge.dest_vertex].append(uc_vertex)
                self.functional_map[edge.src_vertex].append(uc_vertex)

    def forward(self):
        '''This computes all the vertices in the SCM.'''
        for vertex, _ in self.functional_map.items():
            self.recursive_forward(vertex)
           
    def recursive_forward(self, vertex):
        '''This computes the values of a single vertex'''
        if vertex in self.value_map:
            # Return vertex if found in the map
            return self.value_map[vertex]
        input_value_list = []
        for input_vertex in self.functional_map[vertex]:
            input_value_list.append(self.recursive_forward(input_vertex))
        # Now concatenate the input tensors along the Channel dimension
        self.value_map[vertex] = torch.cat(input_value_list, dim=1)
        return self.value_map[vertex]
    
    def clear_intermediate_values(self):
        '''Clears all the values accrued during the forward pass'''
        self.value_map = {}
            
            
class SCMFunction(nn.Module):
    '''Represents the functional relationships for an SCM'''
    def __init__(self, input_dims, output_classes):
        super().__init__()
        C,H,W = input_dims
        CONV_KERNEL_SIZE = 3
        POOLING_KERNEL_SIZE = 2
        POOLING_STRIDE = 2
        CNN1_CHANNELS = 32
        CNN2_CHANNELS = 16
        CNN3_CHANNELS = 1
        # Calculate what the dimensions would look like for the final layer.
        new_dims = self.calculate_conv_dims(input_dims, CNN1_CHANNELS, CONV_KERNEL_SIZE)
        new_dims = self.calculate_conv_dims(new_dims, new_dims[0], POOLING_KERNEL_SIZE, stride=POOLING_STRIDE)
        new_dims = self.calculate_conv_dims(new_dims, CNN2_CHANNELS, CONV_KERNEL_SIZE)
        new_dims = self.calculate_conv_dims(new_dims, new_dims[0], POOLING_KERNEL_SIZE, stride=POOLING_STRIDE)
        new_dims = self.calculate_conv_dims(new_dims, CNN3_CHANNELS, CONV_KERNEL_SIZE)
        linear_input_features = math.prod(new_dims)
        # instantiate the layers we need.
        self.cnn1 = nn.Conv2d(in_channels=C, out_channels=CNN1_CHANNELS, kernel_size=CONV_KERNEL_SIZE)
        self.cnn2 = nn.Conv2d(in_channels=CNN1_CHANNELS, out_channels=CNN2_CHANNELS, kernel_size=CONV_KERNEL_SIZE)
        self.cnn3 = nn.Conv2d(in_channels=CNN2_CHANNELS, out_channels=CNN3_CHANNELS, kernel_size=CONV_KERNEL_SIZE)
        self.linear = nn.Linear(linear_input_features, output_classes)
        self.pooling = nn.MaxPool2d(POOLING_KERNEL_SIZE, POOLING_STRIDE)
        self.relu = nn.ReLU()
                    
    def calculate_conv_dims(self, input_dims, out_channels, kernel_size, stride=1, padding=0):
        C,H,W = input_dims
        new_c = out_channels
        new_h = (H - kernel_size + 2*padding) / stride + 1 
        new_w = (W - kernel_size + 2*padding) / stride + 1
        return (new_c, new_h, new_w)
    
    def forward(self, x, intervention=None):
        if intervention is not None:
            return intervention
        x = self.cnn1(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
        



