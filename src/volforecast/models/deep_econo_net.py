import torch
import torch.nn as nn
import torch.optim as optim


class DeepEconoNet(nn.Module):
    def __init__(self, seq_len=20, lr=1e-3, device=None):
        super().__init__()
        
        # Use GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ----- Model Layers -----
        self.conv = nn.Conv1d(
            in_channels=1,     # one log-return per day
            out_channels=16,   # number of filters
            kernel_size=3,
            padding=1
        )

        self.lstm = nn.LSTM(
            input_size=16,     # must match conv filters
            hidden_size=32,
            num_layers=1,      # dropout ignored for 1-layer LSTM
        )

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

        # ----- Loss & Optimizer -----
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.seq_len = seq_len
        self.to(self.device)


    # ---------- Forward Pass ----------
    def forward(self, x):
        """
        x: (batch, seq_len, 1)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)        # (batch, 1, seq_len)
        x = torch.relu(self.conv(x)) # (batch, 16, seq_len)

        # LSTM expects (seq_len, batch, features)
        x = x.transpose(1, 2)        # (batch, seq_len, 16)
        x = x.transpose(0, 1)        # (seq_len, batch, 16)

        lstm_out, (h, c) = self.lstm(x)
        h = h[-1]  # last layer’s hidden state: (batch, 32)

        x = torch.relu(self.fc1(h))  # (batch, 16)
        x = self.fc2(x)              # (batch, 1)

        return x


    # ---------- One Training Step ----------
    def train_step(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        self.optimizer.zero_grad()
        preds = self.forward(X_batch)
        loss = self.criterion(preds, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    # ---------- Full Training Loop ----------
    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True):
        """
        train_loader: DataLoader yielding (X, y)
        val_loader: optional DataLoader
        """
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0
            batches = 0

            for X_batch, y_batch in train_loader:
                loss = self.train_step(X_batch, y_batch)
                total_loss += loss
                batches += 1

            avg_train_loss = total_loss / batches

            # ----- Validation -----
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    vbatches = 0
                    for Xv, yv in val_loader:
                        Xv = Xv.to(self.device)
                        yv = yv.to(self.device)
                        preds = self.forward(Xv)
                        loss_v = self.criterion(preds, yv)
                        val_loss += loss_v.item()
                        vbatches += 1

                avg_val_loss = val_loss / vbatches

                if verbose:
                    print(f"Epoch {epoch}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}")

            else:
                if verbose:
                    print(f"Epoch {epoch}: train={avg_train_loss:.6f}")
