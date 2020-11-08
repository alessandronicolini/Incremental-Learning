import numpy as np
import pickle

class InfoLog():


    def __init__(self, run, print_info=True, saving_folder=None):
        
        self._saving_folder = saving_folder
        self._print_info = print_info
        
        self._run = run
        self._batch_counter = -1
        self._epoch_counter = -1

        self._batch_info = None
        self.run_info = []

        # print on console if requested
        if self._print_info:
            print("RUN %i ------------------------------------------------------------------------------------\n" % \
                (self._run), end='\n')


    def new_class_batch(self):
   
        # increment batch counter
        self._batch_counter += 1

        # initialize a new batch info container, a dictionary object
        self._batch_info = {
            'train_acc':[],
            'train_loss':[],
            'val_acc':[],
            'val_loss':[],
            'best': {'val_acc':None, 'val_loss': np.inf, 'epoch_num':None, 'state_dict':None},
            'test_acc': None
        }
        
        # initialize epoch counter and _epoch_info dict
        self._epoch_counter = -1

        # print on console if requested
        if self._print_info:
            print("\tCLASS BATCH %i\n" % (self._batch_counter), end='\n')
    

    def new_epoch(self):

        # increment epoch counter
        self._epoch_counter += 1


    def store_info(self, train_acc=None, train_loss=None, val_acc=None, val_loss=None, 
    test_acc=None, state_dict=None):

        if train_acc!=None and train_loss!=None and val_acc!=None and val_loss!=None and state_dict!=None:
            self._batch_info['train_acc'].append(train_acc)
            self._batch_info['train_loss'].append(train_loss)
            self._batch_info['val_acc'].append(val_acc)
            self._batch_info['val_loss'].append(val_loss)
            
            # update best info
            if val_loss < self._batch_info['best']['val_loss']:
                self._batch_info['best']['val_acc'] = val_acc
                self._batch_info['best']['val_loss'] = val_loss
                self._batch_info['best']['epoch_num'] = self._epoch_counter
                self._batch_info['best']['state_dict'] = state_dict
                
            # print train_acc, train_loss, val_acc, val_loss
            if self._print_info:
                print("\tepoch %2i:    train_acc: %.3f  train_loss: %.3f  val_acc: %.3f  val_loss: %.3f" % \
                    (self._epoch_counter, 
                     self._batch_info['train_acc'][-1], 
                     self._batch_info['train_loss'][-1], 
                     self._batch_info['val_acc'][-1], 
                     self._batch_info['val_loss'][-1]), end='\n')

        elif test_acc != None:
            self._batch_info['test_acc'] = test_acc
            
            # update run info
            self.run_info.append(self._batch_info)
            
            # print test acc and best val loss
            if self._print_info:
                print("\n\tBEST RESULTS:\tval_loss: %.3f,  val_acc: %.3f,  epoch:git push  %i" % \
                    (self._batch_info['best']['val_loss'],
                     self._batch_info['best']['val_acc'], 
                     self._batch_info['best']['epoch_num']), end='\n')
                print("\tTEST ACC: %.3f\n\n" % (self._batch_info['test_acc']))
            
            # save run information to file
            if self._saving_folder != None:
                saving_path = self._saving_folder+"/run_"+str(self._run)+".pkl"
                self._to_file(saving_path)
        
        
    def _to_file(self, saving_path):

        with open(saving_path, "wb") as file:
            pickle.dump(self.run_info, file, pickle.HIGHEST_PROTOCOL)