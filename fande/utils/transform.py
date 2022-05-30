

def get_vectors_e(tensor_X, tensor_Y, n_molecules, n_atoms):

      tensor_X = tensor_X.view(3*n_atoms+1,n_molecules,-1).transpose(0,1)
      vectors_X = tensor_X[:, -1, :].squeeze()

      tensor_Y = tensor_Y.view(3*n_atoms+1,-1).transpose(0,1)
      vector_Y = tensor_Y[:,-1].squeeze()
      
      return vectors_X, vector_Y

def get_vectors_f(tensor_X, tensor_Y, n_molecules, n_atoms):

      tensor_X = tensor_X.view(3*n_atoms+1,n_molecules,-1).transpose(0,1)
      vectors_X = tensor_X[:,:-1, :].squeeze()

      tensor_Y = tensor_Y.view(3*n_atoms+1,-1).transpose(0,1)
      vector_Y = tensor_Y[:-1,:].squeeze()
      
      return vectors_X, vector_Y

def extract_energies(forces_energies):
      res_ = forces_energies[:,-1]
      return res_

def extract_forces(forces_energies):
      res_ = forces_energies[:,:-1]
      return res_

