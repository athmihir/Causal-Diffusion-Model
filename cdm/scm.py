from torch import nn
import torch
import math
from enum import Enum
from collections import defaultdict
from cdm.constants import IMAGE_CHANNELS, IMAGE_RESOLUTION

class EdgeType(Enum):
    DIRECTED = 1
    BI_DIRECTED = 2

class Vertex():
    def __init__(self, label, dimensions):
        self.label = label
        self.dimensions = dimensions

    def __hash__(self):
        return hash((self.label))
    
    def __eq__(self, other):
        return self.label == other.label and self.dimensions == other.dimensions
    
    def __str__(self):
        return f"Vertex(label:'{self.label}, dimensions: '{self.dimensions}'')"
    
    def is_unobserved(self):
        if self.label.startswith("U_"):
            return True
        return False


class Edge():
    def __init__(self, src_vertex:Vertex, dest_vertex:Vertex, edge_type=EdgeType):
        self.src_vertex = src_vertex
        self.dest_vertex = dest_vertex
        self.edge_type = edge_type

class SCM(nn.Module):
    def __init__(self, edges: list[Edge]):
        super().__init__()
        # Holds the weights for each edge in the SCM
        self.layers = nn.ModuleDict()
        # Holds the weights for the Autoencoder layers.
        self.ae_layers = nn.ModuleDict()
        # Holds the main DAG adjacency list
        self.functional_map = defaultdict(list)
        # Holds the computed value tensors for the forward passes.
        self.value_map = {}
        # Holds the output tensors for VertexAE 
        self.value_map_2 = {}
        # construct the functional map 
        self.construct_functional_map(edges)
        # define the ncm functional components for each vertex in the functional map
        self.define_functional_components()
        
    
    def define_functional_components(self):
        '''Defines the NCM functional relationships between input and output vertex'''
        for vertex, input_list in self.functional_map.items():
            # ie. vertex does not represent exogenous U.
            if not vertex.is_unobserved():
                # These are the total channels after combining all inputs to the node.
                # We start with 3 because each endogenous V also gets the original image as a prior.
                total_channels = 3
                for element in input_list:
                    total_channels += element.dimensions[0]
                self.layers[str(vertex)] = SCMFunction((total_channels, IMAGE_RESOLUTION, IMAGE_RESOLUTION), output_classes=vertex.dimensions[-1])
                self.ae_layers[str(vertex)] = VertexAE(vertex.dimensions[-1], vertex.dimensions[-1])

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
                self.functional_map[edge.dest_vertex].append(uc_vertex)
                self.functional_map[edge.src_vertex].append(uc_vertex)
        # Append unobserved confounding as dependency for those V that do not have an unobserved yet assigned to them.
        for vertex, _ in self.functional_map.items():
            if not vertex.is_unobserved() and not self.has_unobserved(vertex):
                uc_vertex = Vertex(f"U_{vertex.label}", dimensions=(IMAGE_CHANNELS, IMAGE_RESOLUTION, IMAGE_RESOLUTION))
                self.functional_map[vertex].append(uc_vertex)

    def forward(self, I, intervention_dict:dict[Vertex, torch.tensor]=None):
        '''This computes all the vertices in the SCM.'''
        for vertex, _ in self.functional_map.items():
            self.recursive_forward(vertex, I, intervention_dict)
        return self.value_map, self.value_map_2
           
    def recursive_forward(self, vertex, I, intervention_dict=None):
        '''This computes the values of a single vertex'''
        # Return vertex if found in the map
        if vertex in self.value_map:
            return self.value_map[vertex]
        # If we are asked to perform an intervention, then we just set it
        if intervention_dict is not None and vertex in intervention_dict:
            self.value_map[vertex] = self.ae_layers[str(vertex)].decode(intervention_dict[vertex])
            self.value_map_2[vertex] = (intervention_dict[vertex], self.value_map[vertex])
        # If this is a exogenous vertex, we can simply set it as random
        elif vertex.label.startswith("U_"):
            self.value_map[vertex] = torch.randn(tuple([I.shape[0]] + list(vertex.dimensions)), device=I.device)
        else:
            # Since the image will always be an input fed into the SCM... 
            input_value_list = [I]
            for input_vertex in self.functional_map.get(vertex):
                input_value = self.recursive_forward(input_vertex, I, intervention_dict)
                # Broadcast the shape to ensure the correct dimensions for input are maintained
                if len(input_value.shape) < 3:
                    input_value = input_value.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, IMAGE_RESOLUTION, IMAGE_RESOLUTION)
                input_value_list.append(input_value)
            # Concatenate all the input tensors and pass it through the deriving function for the vertex
            self.value_map[vertex] = self.layers[str(vertex)](torch.cat(input_value_list, dim=1))
            # Also autoencode the vertex.
            self.value_map_2[vertex] = self.ae_layers[str(vertex)](self.value_map[vertex])
        return self.value_map[vertex]
    
    def clear_intermediate_values(self):
        '''Clears all the values accrued during the forward pass'''
        self.value_map = {}
        self.value_map_2 = {}
    
    def clear_endogenous_values(self):
        '''Clears only the endogenous values, but keeps the exogenous values. Useful during interventions and recomputing the SCM.'''
        clear_list = list(filter(lambda v: not v.is_unobserved() , self.value_map))
        for element in clear_list:
            self.value_map.pop(element)
            self.value_map_2.pop(element)


    def has_unobserved(self, vertex):
        for c_v in self.functional_map.get(vertex):
            if c_v.label.startswith("U_"):
                return True
        return False

class VertexAE(nn.Module):
    '''This AE encodes high dimensional V into low dimensional class labels. 
    It also decodes low dimensional class labels into high dimensional V.'''
    def __init__(self, input_dims, output_classes):
        super().__init__()
        hidden_dims = 50
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dims, hidden_dims),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dims, output_classes),
            nn.Softmax(dim=1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_classes, hidden_dims ),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dims, hidden_dims),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dims, input_dims),
            nn.SiLU(),  # Swish activation function
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        '''Takes V(high_dim), returns V(low_dim), V(high_dim_reconstructed),'''
        v_low = self.encode(x)
        v_high_reconstructed = self.decode(v_low)
        return (v_low, v_high_reconstructed)
    

class SCMFunction(nn.Module):
    '''Represents the functional relationships for an SCM'''
    def __init__(self, input_dims, output_classes):
        super().__init__()
        C,H,W = input_dims
        CONV_KERNEL_SIZE = 3
        CONV_PADDING = 1
        POOLING_KERNEL_SIZE = 2
        POOLING_STRIDE = 2
        CNN1_CHANNELS = 32
        CNN2_CHANNELS = 16
        CNN3_CHANNELS = 1
        # Calculate what the dimensions would look like for the final layer.
        new_dims = self.calculate_conv_dims(input_dims, CNN1_CHANNELS, CONV_KERNEL_SIZE, padding=CONV_PADDING)
        new_dims = self.calculate_conv_dims(new_dims, new_dims[0], POOLING_KERNEL_SIZE, stride=POOLING_STRIDE)
        new_dims = self.calculate_conv_dims(new_dims, CNN2_CHANNELS, CONV_KERNEL_SIZE, padding=CONV_PADDING)
        new_dims = self.calculate_conv_dims(new_dims, new_dims[0], POOLING_KERNEL_SIZE, stride=POOLING_STRIDE)
        new_dims = self.calculate_conv_dims(new_dims, CNN3_CHANNELS, CONV_KERNEL_SIZE, padding=CONV_PADDING)
        linear_input_features = math.prod(new_dims)
        # instantiate the layers we need.
        self.cnn1 = nn.Conv2d(in_channels=C, out_channels=CNN1_CHANNELS, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING)
        self.cnn2 = nn.Conv2d(in_channels=CNN1_CHANNELS, out_channels=CNN2_CHANNELS, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING)
        self.cnn3 = nn.Conv2d(in_channels=CNN2_CHANNELS, out_channels=CNN3_CHANNELS, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING)
        self.linear = nn.Linear(linear_input_features, output_classes)
        self.pooling = nn.MaxPool2d(POOLING_KERNEL_SIZE, POOLING_STRIDE)
        self.relu = nn.ReLU()
                    
    def calculate_conv_dims(self, input_dims, out_channels, kernel_size, stride=1, padding=0):
        C,H,W = input_dims
        new_c = out_channels
        new_h = (H - kernel_size + 2*padding) / stride + 1 
        new_w = (W - kernel_size + 2*padding) / stride + 1
        return (new_c, int(new_h), int(new_w))
    
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
        



