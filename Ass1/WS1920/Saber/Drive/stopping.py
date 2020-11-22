import math
from trainer import Trainer

class EarlyStoppingCallback:
    def __init__(self, patience):
        #initialize all members you need
        self.patience = patience
        self.valid_losses = []
        self.no_progress_count = 0
        self.loss_list = []
        self.s = False

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress

        print("loss=", current_loss)
        self.loss_list.append(current_loss)
        self.valid_losses.append(current_loss)
        best_loss = min(self.valid_losses)

        if current_loss <= best_loss:
            self.s = True
            self.no_progress_count = 0
        else:
            self.s = False
            self.no_progress_count += 1


        return self.should_stop()

    def should_stop(self):

        # check whether the duration of where there was no progress is larger or equal to the patience

        should_stop = False

        if self.no_progress_count > self.patience:
            should_stop = True
            # self.no_progress_count = 0
            self.valid_losses.clear()
            print ("losses=",self.loss_list)

        return should_stop



