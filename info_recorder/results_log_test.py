from results_log import ResultsLog
import os

folder = "test_results"
os.mkdir(folder)

log = ResultsLog()

# cycle on runs
for i in range(3):
    # make the log know a new run starts
    log.new_run() 
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
            log.update_epochs_info(train_acc=1, val_acc=2, train_loss=3, val_loss=4, model_params=7) 
        
        """
            test the best model
        """  
        # add test accuracy
        log.update_epochs_info(test_acc=5)    
        # update batches info before starting a new one
        log.update_batches_info()  
    # update runs before starting the new run
    log.update_runs_info()

log.to_file(folder)
print(log.runs_info)