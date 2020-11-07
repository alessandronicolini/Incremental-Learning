import numpy as np
import pickle

"""
TO DO: 
    - in the benchmark class create a directory for the results of the current method 
    - add training time information
"""

class InfoLog():


    def __init__(self, num_runs=3, num_batches=10, num_epochs=15, print_info=True):
        self._print_info = print_info
        self._num_runs = num_runs
        self._num_batches = num_batches
        self._num_epochs = num_epochs
        self._run_counter = -1
        self._batch_counter = -1
        self._epoch_counter = -1
        self._batches_info = None
        self._epochs_info = None
        self.runs_info = dict()


    def new_run(self):
        """
            updates current run index and inizialize the class batch index
        """

        # increment runs counter and set up the corrisponding element od runs_info dict
        self._run_counter += 1
        self.runs_info[self._run_counter] = None

        # initialize batch counter and _batches_info dict
        self._batch_counter = -1
        self._batches_info = dict()

        # print on console if requested
        if self._print_info:
            print("RUN %i ------------------------------------------------------------------------------------\n" % \
                (self._run_counter), end='\n')


    def new_class_batch(self):
        """
            updates the current class batch index: starts from 0 up to 9.
        """ 

        # increment batch counter and set up the corrisponding element on _batch_infor dict
        self._batch_counter += 1
        self._batches_info[self._batch_counter] = None
        
        # initialize epoch counter and _epoch_info dict
        self._epoch_counter = -1
        self._epochs_info = {
            'train_acc':[],
            'train_loss':[],
            'val_acc':[],
            'val_loss':[],
            'best': {'val_loss': np.inf, 'epoch_num':None, 'state_dict':None},
            'test_acc': None
        }

        # print on console if requested
        if self._print_info:
            print("\tCLASS BATCH %i\n" % (self._batch_counter), end='\n')
    
    def new_epoch(self):
        """
            update current epoch index
        """

        # increment epoch counter
        self._epoch_counter += 1


    def store_info(self, train_acc=None, train_loss=None, val_acc=None, val_loss=None, 
    test_acc=None, state_dict=None):
        """
            updates current class batch info, use it:
            - in the epoch cycle: to record train_acc, val_loss, val_acc, val_loss and to update
              best_val_loss and related model params (should be a dict)
            - at the end of epochs: to record test_acc
        """

        if train_acc!=None and train_loss!=None and val_acc!=None and val_loss!=None and state_dict!=None:
            self._epochs_info['train_acc'].append(train_acc)
            self._epochs_info['train_loss'].append(train_loss)
            self._epochs_info['val_acc'].append(val_acc)
            self._epochs_info['val_loss'].append(val_loss)
            
            # update best info
            if val_loss < self._epochs_info['best']['val_loss']:
                self._epochs_info['best']['val_loss'] = val_loss
                self._epochs_info['best']['epoch_num'] = self._epoch_counter
                self._epochs_info['best']['state_dict'] = state_dict
            
            # print train_acc, train_loss, val_acc, val_loss
            if self._print_info:
                print("\tepoch %2i:    train_acc: %.3f  train_loss: %.3f  val_acc: %.3f  val_loss: %.3f" % \
                    (self._epoch_counter, 
                     self._epochs_info['train_acc'][-1], 
                     self._epochs_info['train_loss'][-1], 
                     self._epochs_info['val_acc'][-1], 
                     self._epochs_info['val_loss'][-1]), end='\n')

        elif test_acc != None:
            self._epochs_info['test_acc'] = test_acc
            
            # check for updating batch and run info
            self._batches_info[self._batch_counter] = self._epochs_info
            
            if self._batch_counter == self._num_batches-1:
                self.runs_info[self._run_counter] = self._batches_info
            
            # print test acc and best val loss
            if self._print_info:
                print("\n\tBEST VAL LOSS is %.3f in epoch %i" % \
                    (self._epochs_info['best']['val_loss'], 
                     self._epochs_info['best']['epoch_num']), end='\n')
                print("\tTEST ACC: %.3f\n\n" % (self._epochs_info['test_acc']))
        
        
    def to_file(self, folder):
        """
            save registered info as a pickle file
        """
        with open(folder+"/runs_info.pkl", "wb") as file:
            pickle.dump(self.runs_info, file, pickle.HIGHEST_PROTOCOL)
    

