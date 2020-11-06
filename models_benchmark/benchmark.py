class Benchmark():
    
    def __init__(self, num_runs, num_class_batches, num_epochs, batch_size, dataloaders,
     seeds, model, criterion, optimizer, scheduler, resultsLog):
        self.num_epochs = num_epochs
        self.num_runs = num_runs
        self.num_class_batches = num_class_batches
        self.batch_size = batch_size
        self.dataloaders = dataloaders
        self.seeds = seeds
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log = resultsLog(num_runs=num_runs, num_batches=num_class_batch, num_epochs=n_epochs)
        
    
    def _train_batch(self, data_batch):
        pass

    def _train_epoch(self):
        pass

    def _validate(self):
        pass

    def _train(self):
        pass

    def _test(self):
        pass

    def execute(self):
        
        # cycle for each run
        for run in self.num_runs:
            self.log.new_run() # update log internal state
            
            # cycle for each class_batch 
            for class_batch in self.num_class_batches:
                self.log.new_class_batch() # update log internal state
                
                # cycle for each epoch
                for epoch in self.num_epochs:
                    self.log.new_epoch() # update log internal state
                    pass 



    def plot_results(self):
        pass