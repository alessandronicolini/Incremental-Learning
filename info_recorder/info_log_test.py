from info_log import InfoLog
import os

folder = "test_results"
try:
    os.mkdir(folder)
except FileExistsError:
    pass

# cycle on runs
for i in range(3):
    
    log = InfoLog(run=i, saving_folder=folder) 
    # cycle on class batches
    for j in range(10):
        # make the log know a new batch of classes starts
        log.new_class_batch() 
        # cycle on epochs 
        for k in range(15):
            # make the log know a new epoch starts
            log.new_epoch()
            
            """
                train and validate the network
                ...compute info here...
            """

            # update _epochs_info
            log.store_info(train_acc=1, val_acc=2, train_loss=3, val_loss=4, state_dict=7) 
        
        """
            test the best model
        """  
        # add test accuracy
        log.store_info(test_acc=5)    