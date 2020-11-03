from results_log import ResultsLog

log = ResultsLog()

# cycle on runs
for i in range(3):
    
    log.new_run() # make the log know a new run started
    
    # cycle on class batches
    for j in range(10):
        
        log.new_class_batch() # make the log know a new batch of classes started
        
        # cycle on epochs 
        for k in range(15):
            log.new_epoch()
            # train and validate the network
            log.add_info(train_acc=1, val_acc=2, train_loss=3, val_loss=4, model_params=7) 
        
        log.add_info(test_acc=5)
        log.commit_batch_info()
