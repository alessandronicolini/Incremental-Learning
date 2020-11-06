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

    def _train_epoch(self, run, class_batch):
        number_of_elements = None
        running_loss = 0
        running_corrects = 0
        for data_batch in dataloaders[run][class_batch]['train']:
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
            
            # update log internal state
            self.log.new_run() 
            
            # cycle for each class_batch 
            for class_batch in self.num_class_batches:
                
                # update log internal state
                self.log.new_class_batch() 
            
                # cycle for each epoch
                for epoch in self.num_epochs:
                    
                    # update log internal state
                    self.log.new_epoch() 
                    
                    pass 



    def plot_results(self):
        pass