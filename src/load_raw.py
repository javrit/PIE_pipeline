#Création d'un fichier permettant de load le fichier "driving_sample.raw" 
#Cela ne permet pas de traiter les données en temps réel

from aestream import FileInput

def load_raw(file_path : str, shape : tuple, device : str = "cpu"):
    # Open a file while specifying it's shape (shape[0], shape[1]) and its device ("cpu")
    # By default, we send the tensors to the CPU with Numpy
    #   - if you have a PyTorch installation with a GPU, try changing this to "cuda"
    
    events = FileInput(file_path, shape, device).load()
    
    return events