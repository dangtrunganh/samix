class TaskAccuracy:

    def __init__(self, task_index, num_epochs):
        self.task_index = task_index
        self.num_epochs = num_epochs

        # class-il
        self.last_acc = 0.0
        self.best_acc = (0, 0.0)
        self.dict_acc_classes = {}

        # task-il
        self.task_acc = 0.0
        self.best_task_acc = (0, 0.0)
