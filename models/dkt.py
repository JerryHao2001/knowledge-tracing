import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.optim import Adam
from sklearn import metrics

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor, BoolTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor, LongTensor, BoolTensor


class DKT(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=False
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        qr = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(qr))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(
        self, questions, responses, targets, deltas, masks,
        train_config, pad_val=-1e+3
    ):
        batch_size = train_config["batch_size"]
        num_epochs = train_config["num_epochs"]
        train_ratio = train_config["train_ratio"]
        learning_rate = train_config["learning_rate"]

        train_idx = int(len(questions) * train_ratio)

        train_questions = questions[:train_idx]
        train_responses = responses[:train_idx]
        train_targets = targets[:train_idx]
        train_deltas = deltas[:train_idx]
        train_masks = masks[:train_idx]

        test_questions = questions[train_idx:]
        test_responses = responses[train_idx:]
        test_targets = targets[train_idx:]
        test_deltas = deltas[train_idx:]
        test_masks = masks[train_idx:]

        opt = Adam(self.parameters(), learning_rate)

        aucs = []
        loss_means = []

        for i in range(1, num_epochs + 1):
            loss_mean = []
            for _ in range(train_idx // batch_size):
                random_indices = np.random.choice(
                    train_idx, batch_size, replace=False
                )

                q = train_questions[random_indices]
                r = train_responses[random_indices]
                t = train_targets[random_indices]
                d = train_deltas[random_indices]
                m = train_masks[random_indices]

                self.train()

                y = self(LongTensor(q), LongTensor(r))
                y = (y * one_hot(LongTensor(d), self.num_q)).sum(-1)

                y = torch.masked_select(y, BoolTensor(m))
                t = torch.masked_select(FloatTensor(t), BoolTensor(m))

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            self.eval()

            test_y = self(
                LongTensor(test_questions), LongTensor(test_responses)
            )
            test_y = (
                test_y * one_hot(LongTensor(test_deltas), self.num_q)
            ).sum(-1)
            test_y = torch.masked_select(test_y, BoolTensor(test_masks))\
                .detach().cpu()

            test_t = torch.masked_select(
                LongTensor(test_targets), BoolTensor(test_masks)
            ).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=test_t.numpy(), y_score=test_y.numpy()
            )

            loss_mean = np.mean(loss_mean)

            print(
                "Epoch: {},   AUC: {},   Loss Mean: {}"
                .format(i, auc, loss_mean)
            )

            aucs.append(auc)
            loss_means.append(loss_mean)

        return aucs, loss_means
