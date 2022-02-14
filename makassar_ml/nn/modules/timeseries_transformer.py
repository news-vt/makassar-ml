from __future__ import annotations
import torch
from . import Time2Vec


class TimeseriesTransformer(torch.nn.Module):

    def __init__(self,
        n_input_features: int,
        n_output_features: int,
        d_time_embed: int,
        d_model: int = 512,
        dropout: float = 0.1,
        batch_first: bool = False,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_encoder_heads: int = 8,
        n_decoder_heads: int = 8,
        ):
        super().__init__()

        self.n_input_features = n_input_features
        self.n_output_features = n_output_features
        self.d_time_embed = d_time_embed
        self.d_model = d_model
        self.dropout = dropout
        self.batch_first = batch_first
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_encoder_heads = n_encoder_heads
        self.n_decoder_heads = n_decoder_heads

        # Time embedding.
        self.time_projection = Time2Vec(input_dim=n_input_features, embed_dim=d_time_embed)

        # Linear transformation from input-feature space into arbitrary n-dimension space.
        # This is similar to a word embedding used in NLP tasks.
        self.encoder_projection = torch.nn.Linear(in_features=d_time_embed, out_features=d_model)
        self.decoder_projection = torch.nn.Linear(in_features=n_output_features, out_features=d_model)

        # Transformer encoder/decoder layers.
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_encoder_heads, # Number of multihead-attention models.
            dropout=dropout,
            dim_feedforward=4*d_model,
            batch_first=batch_first,
        )
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_decoder_heads, # Number of multihead-attention models.
            dropout=dropout,
            dim_feedforward=4*d_model,
            batch_first=batch_first,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers)

        # Linear output layer.
        # We typically only predict a single data point at a time, so output features is typically 1.
        self.linear = torch.nn.Linear(in_features=d_model, out_features=n_output_features)

    def encode(self, src: torch.Tensor) -> torch.Tensor:

        # Embed the source into time-feature dimensions.
        x = self.time_projection(src)

        # Transform time embedding into arbitrary feature space
        # for the attention encoder model.
        x = self.encoder_projection(x)

        # Pass the linear transformation through the encoder layers.
        x = self.encoder(x)

        return x

    def decode(self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        ) -> torch.Tensor:
        """Decode function.

        Args:
            tgt (torch.Tensor): The sequence to the decoder
            memory (torch.Tensor): The sequence from the last layer of the encoder

        Returns:
            torch.Tensor: Decoded sequence.
        """
        # Transform target into arbitrary feature space.
        x = self.decoder_projection(tgt)

        # Pass the linear transformation through the decoder layers.
        x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)

        # Pass the output of the decoder through the linear prediction layer.
        x = self.linear(x)
        return x

    def forward(self, 
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        ) -> torch.Tensor:
        x = self.encode(src)
        y = self.decode(tgt=tgt, memory=x, tgt_mask=tgt_mask)
        return y