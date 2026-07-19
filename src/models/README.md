FactoryShield-OT

Component: Deep Learning Model Training Pipeline
Files: src/models/lstm_autoencoder.py | src/models/train.py
Dataset Target: HAIEnd 23.05 (ICS/OT End-device behavior)

-WHAT IT IS

This is the brain of the anomaly detection system. It uses an LSTM Autoencoder to learn the absolute "normal" thermodynamic behavior of the Emerson Ovation DCS boiler process.

It works on a semi-supervised principle: we only show it clean, healthy data during training. Once deployed, anything it fails to reconstruct accurately is flagged as a potential cyberattack.

-THE MATH & ARCHITECTURE (lstm_autoencoder.py)

1. The Sliding Window (SlidingWindowDataset)

Industrial data is a continuous stream. LSTMs need time-bounded context. This class slices the (N, 225) continuous feature array into overlapping (T=60, 225) blocks.

The Geeky Part: Zero-copy tensor views.
Instead of duplicating RAM, we just slide a pointer over the raw data.

    def __getitem__(self, idx):
        start = idx * self.stride
        return self.data[start:start + self.window]


2. The Compression (LSTMEncoder)

Takes 60 seconds of reality across 225 sensors and squeezes it through an LSTM down to a tight, dense vector (Latent Dimension = 32). This forces the model to learn the most important physics correlations.

The Geeky Part: Taking the final hidden state.
We ignore the intermediate LSTM outputs and only keep the absolute last hidden state (h_n[-1]), squishing it into our 32-dim latent space.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> Batch, Time, Features
        _, (h_n, _) = self.lstm(x)
        return self.to_latent(h_n[-1])  # (B, latent_dim)


3. The Reconstruction (LSTMDecoder)

Takes that compressed 32-dim vector and tries to blow it back up into the original 60-second, 225-sensor sequence.

The Geeky Part: Seeding the timeline.
We take the latent vector z, expand it back to hidden_size, and then physically duplicate it 60 times (repeat(1, self.seq_len, 1)) to force the LSTM to unroll time.

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Replicate Z across T as decoder input
        seed = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(seed)
        return self.to_features(out)  # (B, T, F)


THE PIPELINE (train.py)

This script handles the heavy lifting of the optimization loop.

Metric: L1Loss (Mean Absolute Error).
We calculate the difference between the real window (batch) and the reconstructed window (recon).

    recon = model(batch)
    loss = criterion(recon, batch)  # nn.L1Loss()
    loss.backward()
    optimizer.step()


Checkpointing: It monitors the train_mae. It only overwrites models_saved/lstm_autoencoder_best.pth when a strict improvement in the MAE is detected.

-RUNNING IT ON POTATO HARDWARE

LSTMs are brutal on CPUs. The dataset is massive (896k+ rows). If you are running this locally without a CUDA GPU, use the --stride parameter in train.py.

Default: --stride 1 (Advances 1 second at a time)

Fast mode: --stride 10 (Advances 10 seconds at a time) -> Cuts CPU math load by 90% while keeping context valid.

Execution:

$ python src/models/train.py --train-csv data/processed/train_clean_scaled.csv --stride 10 --epochs 50
