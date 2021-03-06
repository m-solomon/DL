import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, "C:\\Users\\utg_1\\OneDrive\\Documents\\Mohamed's\\Studies\\FAU\\SS20\\DL\\Ass4\\src_to_implement\\checkpoints\\checkpoint_{:03d}.ckp".format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp =t.load( "C:\\Users\\utg_1\\OneDrive\\Documents\\Mohamed's\\Studies\\FAU\\SS20\\DL\\Ass4\\src_to_implement\\checkpoints\\checkpoint_{:03d}.ckp".format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        #TODO
        ###################################################################################################
        ###################################################################################################
        self._optim.zero_grad()   # reset the gradients
        out = self._model(x)      # propagate through the network
        loss = self._crit(out, y) # calculate the loss
        loss.backward()           # compute gradient by backward propagation
        self._optim.step()        # update weights
        return loss               # return the loss
        ###################################################################################################
        ###################################################################################################
        
        
    
    def val_test_step(self, x, y):
        #TODO
        ###################################################################################################
        ###################################################################################################
        
        out = self._model(x)       # propagate through the network
        sig_obj = t.nn.Sigmoid()
        pred = sig_obj(out)        # predict:
        pred = t.round(pred)       # calculate the predictions
        loss = self._crit(out, y)  # calculate the loss
        return loss, pred          # return the loss and the predictions
        ###################################################################################################
        ###################################################################################################
        
    def train_epoch(self):
        #TODO
        ###################################################################################################
        ###################################################################################################
        self._model.train()                                     # set training mode

        running_loss = 0.0
        l = self._train_dl.__len__()                            
        for data, label in self._train_dl:                      # iterate through the training set
            if self._cuda == True:                              # transfer the batch to "cuda()" -> the gpu if a gpu is given
                device = t.device("cuda:0")
                data, label = data.to(device), label.to(device)

            loss = self.train_step(data, label)                 # perform a training step
            running_loss += loss.item()                         # calculate the average loss for the epoch 
        return running_loss / l                                 # and return it
        ###################################################################################################
        ###################################################################################################
    
    def val_test(self):
        #TODO
        ###################################################################################################
        ###################################################################################################
        self._model.eval()                                          # set eval mode

        losses = 0
        predictions = []
        l = self._val_test_dl.__len__()

        with t.no_grad():                                           # disable gradient computation

            for data, label in self._val_test_dl:                   # iterate through the validation set
                if (self._cuda == True):                            # transfer the batch to the gpu if given
                    device = t.device("cuda:0")
                    data, label = data.to(device), label.to(device) 

                loss, pred = self.val_test_step(data, label)        # perform a validation step
                losses += loss.item()
                predictions.append(pred)                            # save the predictions and the labels for each batch


        return losses / l                                           # calculate the average loss and return it

        print(predictions)                                          # print the calculated metrics
        
        ###################################################################################################
        ###################################################################################################
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        #TODO
        ###################################################################################################
        ###################################################################################################
        train_losses = []  # create a list for the train 
        val_losses = []    # and validation losses
        epoch = 1          # and create a counter for the epoch 
        ###################################################################################################
        ###################################################################################################
        
        while True:
        #TODO
        ###################################################################################################
        ###################################################################################################
            if epoch > epochs:                        # stop by epoch number
                break
            train_loss = self.train_epoch()           # train for a epoch and then calculate the loss and metrics on the validation set
            val_loss = self.val_test()

            train_losses.append(train_loss)           # append the losses to the respective lists
            val_losses.append(val_loss)

            self.save_checkpoint(epoch)               # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)

            if (self._early_stopping_patience > 0):   # check whether early stopping should be performed using the early stopping criterion 
                break                                 # and stop if so

            epoch += 1
            print(222)

        return train_losses, val_losses           # return the losses for both training and validation
        
        ###################################################################################################
        ###################################################################################################
                    
        
        
        