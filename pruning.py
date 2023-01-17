import torch.nn.utils.prune as prune
import torch
from torchinfo import summary


#Unstructured pruning 
def pruning(model,num_pruning):
    model.train()

    #We get all parameters from convolutional and linear layers
    parameters_to_prune = ()
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune += ((module,'weight'),)  
        
        elif isinstance(module, torch.nn.Linear):
            parameters_to_prune += ((module,'weight'),)

        
    #We prune 70% of weights with the lowest L1 norm
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.7,
    )

    print("pruning "+str(num_pruning))



def memory_footprint(batch_size,model):
    batch_size = 32
    model_stats = summary(model, input_size=(batch_size, 3, 32, 32),verbose=0).float_to_megabytes(summary(model, input_size=(batch_size, 3, 32, 32),verbose=0).total_params)
    summary_str = float(model_stats)
    print(summary_str)
    return summary_str