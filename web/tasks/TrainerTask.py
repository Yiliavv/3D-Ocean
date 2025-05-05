
from threading import Thread
from src.trainer.transformer import train_transformer


class TrainerTask(Thread):
    def __init__(self,
                 area, seq_len, offset, n_head, num_encoder_layers,
                 num_decoder_layers, learning_rate, dropout, optimizer,
                 batch_size, epochs):
        super().__init__()
        
        self.area = area
        self.seq_len = seq_len
        self.offset = offset
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self):
        model = train_transformer(
            self.area, self.seq_len, self.offset,
            self.epochs, self.batch_size)
        
        return model