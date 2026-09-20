from torch import nn
import torch
import math


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1,
            self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2),
            1
        )
        end = x[:, -1:, :].repeat(
            1,
            math.floor((self.kernel_size - 1) // 2),
            1
        )
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(
            torch.zeros(cycle_len, channel_size),
            requires_grad=True
        )

    def forward(self, index, length):
        index = index.to(self.data.device)
        gather_index = (
            index.view(-1, 1)
            + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.node = configs.node

        # Cycle
        self.cycleQueue = RecurrentCycle(
            cycle_len=self.cycle_len,
            channel_size=self.node
        )

        # Moving Average (Trend extraction)
        if isinstance(configs.moving_avg, int):
            self.kernel_size = [configs.moving_avg]
        else:
            self.kernel_size = configs.moving_avg

        self.layer = nn.Linear(1, len(self.kernel_size))
        self.moving_avg = nn.ModuleList(
            [moving_avg(kernel, stride=1) for kernel in self.kernel_size]
        )

        # Trend predictor (strong)
        self.trend_model = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Periodic modulation scale
        self.mod_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, cycle_index):

        # RevIN
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Trend extraction
        moving_mean = []

        for func in self.moving_avg:
            ma = func(x)
            moving_mean.append(ma.unsqueeze(-1))

        moving_mean = torch.cat(moving_mean, dim=-1)

        moving_mean = torch.sum(
            moving_mean *
            nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),
            dim=-1
        )

        # Forecast components
        y_trend = self.trend_model(moving_mean.permute(0, 2, 1)).permute(0, 2, 1)

        y_cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len,self.pred_len)

        # Modulation Strength
        modulation = 2 * torch.sigmoid(self.mod_scale * y_cycle) - 1

        # Periodically Modulated Trend
        y_trend = y_trend * (1 + modulation)

        # Final Prediction
        y = y_cycle + y_trend

        # RevIN Denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
