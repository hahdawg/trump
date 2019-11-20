import collections
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class Trainer(nn.Module):

    def __init__(
        self,
        num_classes,
        learning_rate=5e-3,
        num_epochs=100000,
        batch_size=256,
        log_interval=100,
    ):
        super(Trainer, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device {self.device}")
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_interval = log_interval

    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @property
    def criterion(self):
        return nn.CrossEntropyLoss()

    def fit(self, X, decoder):
        self.to(self.device)
        X = torch.from_numpy(X).long()
        X = X.to(self.device)
        num_steps = 0
        running_loss = collections.deque(maxlen=self.log_interval)
        for _ in range(self.num_epochs):
            for i in range(0, X.shape[0], self.batch_size):
                y_s = X[i:i + self.batch_size, :-1]
                y_t = X[i:i + self.batch_size, 1:]
                y_hat_t, _ = self.forward(y_s)
                y_hat_t = y_hat_t.reshape(-1, y_hat_t.shape[-1])
                y_t = y_t.reshape(-1,)
                loss = self.criterion(y_hat_t, y_t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())

                if num_steps % self.log_interval == 0:
                    print(num_steps, np.mean(running_loss))
                    print(self.generate_tweet(12, decoder))

                num_steps += 1


class CharLSTM(Trainer):

    def __init__(self, num_chars, hidden_size, num_layers, **kwargs):
        super(CharLSTM, self).__init__(num_classes=num_chars, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_chars, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, num_chars)

    def forward(self, input, hidden=None):
        batch_size = input.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        embedded = self.embedding(input).transpose(0, 1)
        output, hidden = self.rnn(embedded, hidden)
        output = self.decoder(output).transpose(0, 1)
        return output, hidden

    def init_hidden(self, batch_size):
        return (
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device),
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
        )

    def generate_tweet(self, char_start, decoder, num_chars_to_consider=5):
        y_hat = torch.tensor([char_start]).reshape(1, 1).to(self.device)
        tweet = [char_start]
        hidden = None
        softmax = torch.nn.Softmax(dim=0)
        with torch.no_grad():
            for _ in range(185):
                logits, hidden = self.forward(y_hat, hidden)
                p_hat = softmax(logits.flatten()).cpu().numpy()
                n_largest = p_hat.argsort()[-num_chars_to_consider:]
                p_hat[~n_largest] = 0
                p_hat /= p_hat.sum()
                y_hat = np.random.choice(np.arange(len(p_hat)), p=p_hat, size=1)[0]
                tweet.append(y_hat)
                y_hat = torch.tensor([y_hat]).reshape(1, 1).to(self.device)
        tweet = "".join([decoder[i] for i in tweet])
        return tweet
