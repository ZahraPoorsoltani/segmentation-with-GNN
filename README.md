# segmentation-with-GNN-custom-dataset
we use Unet as backbone to extract features then feed to graph neural network. 
at first we desighn gnn with 3 node and weighted edges. to select data for each node we select slice of 3 different patients with the most similarity. for calculate similarity we choose slices with 
similar area of tumour
