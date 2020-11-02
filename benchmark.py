class Benchmark():
    
    def __init__(self, n_epochs, batch_size, original_dataset, model, optimizer, scheduler, 
    results_log):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.original_dataset = original_dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.results_log = results_log
    
    def _train(self):
        pass

    def _validate(self):
        pass

    def _test(self):
        pass

    def execute(self):
        pass

    def plot_results(self):
        pass