from copy import deepcopy
import torch
import os
"""
ERROR
How many classes are you currently using and what is the shape of your output?
Note that class indices start at 0 so your target should contain indices in the range [0, nb_classes-1].
"""
class Benchmark():
    
    def __init__(self, num_epochs, batch_size, dataloaders, device='cuda', saving_folder=None):
        
        # create a saving folder for the current method
        if saving_folder!= None:
            try:
                os.mkdir(saving_folder)
            except FileExistsError:
                pass

        self.saving_folder = saving_folder
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataloaders = dataloaders
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.log = None 


    def set_criterion(self, criterion):
        self.criterion = criterion

    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def set_scheduler(self, scheduler):
        self.scheduler = scheduler


    def set_model(self, model):
        self.model = model.to(self.device)


    def set_infoLog(self, info_log_class, run):
        self.log = info_log_class(saving_folder=self.saving_folder, run=run)
        

    def _do_batch(self, inputs, labels, train=True):
        
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()

        # training phase 
        if train:
            self.model.train()
            
            outputs = self.model(inputs) 
            _, preds = torch.max(outputs, 1) 
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
        
        # validation phase
        else:
            self.model.eval()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs) 
                _, preds = torch.max(outputs, 1) 
                loss = self.criterion(outputs, labels)
        
        return loss.item(), torch.sum(preds == labels.data)
        

    def _do_epoch(self, run, class_batch, train=True):

        number_of_elements = 0
        running_loss = 0
        running_corrects = 0

        if train:
            phase = 'train'
        else:
            phase = 'val'

        for inputs, labels in self.dataloaders[run][class_batch][phase]:
            number_of_elements += len(labels)
            current_mean_loss, current_correct_pred = self._do_batch(inputs, labels, train)
            running_loss += current_mean_loss*inputs.size()[0]
            running_corrects += current_correct_pred
        
        if train:
            self.scheduler.step()

        epoch_loss = running_loss / number_of_elements
        epoch_acc = running_corrects.double() / number_of_elements

        return epoch_loss, epoch_acc

    
    def _test(self, test_dataloader):
        
        self.model.train(False)

        running_corrects = 0
        number_of_elements = 0
        
        for inputs, labels in test_dataloader:
            number_of_elements += len(labels)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).data.item()

        test_acc = running_corrects / number_of_elements
        return test_acc


    def do_class_batch(self, run, class_batch):
        
        self.log.new_class_batch()

        # cycle for each epoch
        for epoch in range(self.num_epochs):
            self.log.new_epoch()
                    
            # get training info
            train_epoch_loss, train_epoch_acc = self._do_epoch(run, class_batch, train=True)
                    
            # get validation info
            val_epoch_loss, val_epoch_acc = self._do_epoch(run, class_batch, train=False) 
                    
            # store info
            self.log.store_info(
                train_acc=train_epoch_acc, 
                train_loss=train_epoch_loss, 
                val_acc=val_epoch_acc,
                val_loss=val_epoch_loss,
                state_dict=deepcopy(self.model.state_dict()))
                
        # load best model state_dict
        best_state_dict = self.log.batch_info['best']['state_dict']
        self.model.load_state_dict(best_state_dict)
                
        # get test dataloader and compute test acc
        test_dataloader = self.dataloaders[run][class_batch]['test']
        test_acc = self._test(test_dataloader)
                
        # update info log
        self.log.store_info(test_acc=test_acc)