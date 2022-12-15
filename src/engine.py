# %%
from tqdm.auto import tqdm

# %%

prog_bar = tgdm()

# %%
def train(model):
    print("Training.........")
    
    global train_itr
    global train_loss_list
    
    prog_bar = tgdm