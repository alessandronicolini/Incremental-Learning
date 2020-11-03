import numpy as np



class ResultsLog():


    def __init__(self, num_run=3, print_info=True):
        self._print_info = print_info
        self._num_run = num_run
        self._current_run = -1
        self._current_batch = -1
        self._current_epoch = -1
        self._batch_info = {'train_acc':[], 'train_loss':[], 'val_acc':[], 'val_loss':[], 'best_val_loss':{'value':np.inf, 'epoch':None}, 'test_acc':None, 'best_model':None}
        self.run_info = {key_run:{} for key_run in range(self._num_run)}


    def new_run(self):
        """
            updates current run index and inizialize the class batch index
        """

        self._current_run += 1
        self._current_batch = -1
        if self._print_info:
            print("RUN %i ------------------------------------------------------------------------------------\n" % (self._current_run), end='\n')


    def new_class_batch(self):
        """
            updates the current class batch index: starts from 0 up to 9.
        """ 

        self._current_batch += 1
        self._current_epoch = -1
        if self._print_info:
            print("\tCLASS BATCH %i\n" % (self._current_batch), end='\n')
    
    def new_epoch(self):
        """
            update current epoch index
        """
        self._current_epoch += 1


    def add_info(self, train_acc=None, train_loss=None, val_acc=None, val_loss=None, 
    test_acc=None, model_params=None):
        """
            updates current class batch info, use it:
            - in the epoch cycle: to record train_acc, val_loss, val_acc, val_loss and to update
              best_val_loss and related model params (should be a dict)
            - at the end of epochs: to record test_acc
        """

        if train_acc!=None and train_loss!=None and val_acc!=None and val_loss!=None and model_params!=None:
            self._batch_info['train_acc'].append(train_acc)
            self._batch_info['train_loss'].append(train_loss)
            self._batch_info['val_acc'].append(val_acc)
            self._batch_info['val_loss'].append(val_loss)
            # update best validation loss
            if val_loss < self._batch_info['best_val_loss']['value']:
                self._batch_info['best_val_loss']['value'] = val_loss
                self._batch_info['best_val_loss']['epoch'] = self._current_epoch
                self._batch_info['best_model'] = model_params
            # print values
            if self._print_info:
                print("\tepoch %2i:    train_acc: %.3f  train_loss: %.3f  val_acc: %.3f  val_loss: %.3f" % (self._current_epoch, self._batch_info['train_acc'][-1], self._batch_info['train_loss'][-1], self._batch_info['val_acc'][-1], self._batch_info['val_loss'][-1]), end='\n')
        
        if test_acc != None:
            self._batch_info['test_acc'] = test_acc
            if self._print_info:
                print("\n\tBEST VAL LOSS is %.3f in epoch %i" % (self._batch_info['best_val_loss']['value'], self._batch_info['best_val_loss']['epoch']), end='\n')
                print("\tTEST ACC: %.3f\n\n" % (self._batch_info['test_acc']))
    
    

    def commit_batch_info(self):
        """
            actually update the global info adding the class batch info just recorded,
            use it at the end of the epoch
        """

        self.run_info[self._current_run][self._current_batch] = self._batch_info
        self._batch_info = {'train_acc':[], 'train_loss':[], 'val_acc':[], 'val_loss':[], 'best_val_loss':{'value':np.inf, 'epoch':None}, 'test_acc':None, 'best_model':None}


    def info2file(self):
        """
            save registered info as a csv file
        """

        pass

