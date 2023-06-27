import torch
import torch.nn

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step= 0
        self._loss=float('inf')
        self._patience=patience
        self.verbose=verbose

    def validate(self,loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verobse:
                    print('early stopping')
                return True
        else:
            self.step = 0
            self.loss = loss
       
        return False

def create_lr(epoch,lr,warm_up_times):
    # return 0.1*lr*(epoch+1)
    return lr*warm_up_times
