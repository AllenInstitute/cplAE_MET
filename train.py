# Dataset class that can load data at different levels of preprocessing, and select
# for different modalities, platforms, etc.

# Model class that can be initialized with different autoencoder branches, where a 
# branch is only loaded if it couples to another branch.

# Loss generating class/function that can be used to easily produce a loss function
# with the desired parameters.

# Training function should have a consistent form across different model types, 
# abstracting away particular of architecture and loss function.
