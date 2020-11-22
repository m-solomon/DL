import math
from src_to_implement.trainer import Trainer


class EarlyStoppingCallback:

    def __init__(self, patience):
        #initialize all members you need
        self.patience = patience
        self.valid_losses = []
        self.no_progress_count = 0

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress

        if len(self.valid_losses) == 0:
            best_loss = current_loss

        if current_loss < best_loss:
            best_loss = current_loss


        self.valid_losses.append(current_loss)
        # best_loss= min(self.valid_losses)



        if current_loss > best_loss:
            self.no_progress_count += 1

        return self.should_stop()

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        if self.no_progress_count > self.patience:
            should_stop = True
            self.no_progress_count = 0
            self.valid_losses.clear()

        return should_stop



