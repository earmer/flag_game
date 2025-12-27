import torch
import torch.nn as nn


ACTION_COUNT = 5
GRID_ID_COUNT = 20


class PPOTransformerPolicy(nn.Module):
    def __init__(
        self,
        width=20,
        height=20,
        max_players=3,
        vector_dim=64,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        mlp_dim=128,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_players = max_players
        self.vector_dim = vector_dim

        self.grid_id_embed = nn.Embedding(GRID_ID_COUNT, vector_dim)
        self.x_embed = nn.Embedding(width, vector_dim)
        self.y_embed = nn.Embedding(height, vector_dim)
        self.flag_embed = nn.Embedding(2, vector_dim)
        self.prison_embed = nn.Embedding(2, vector_dim)
        self.team_embed = nn.Embedding(2, vector_dim)
        self.player_index_embed = nn.Embedding(max_players, vector_dim)
        self.valid_embed = nn.Embedding(2, vector_dim)
        self.token_type_embed = nn.Embedding(2, vector_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vector_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pre_norm = nn.LayerNorm(vector_dim)

        self.action_head = nn.Linear(vector_dim, ACTION_COUNT)
        self.value_head = nn.Linear(vector_dim, 1)

        grid_x, grid_y = self._build_grid_positions(width, height)
        self.register_buffer("grid_x_idx", grid_x, persistent=False)
        self.register_buffer("grid_y_idx", grid_y, persistent=False)

    @staticmethod
    def _build_grid_positions(width, height):
        xs = []
        ys = []
        for y in range(height):
            for x in range(width):
                xs.append(x)
                ys.append(y)
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    def forward(self, grid_ids, player_features):
        """
        grid_ids: (B, H, W) int64
        player_features: (B, 2 * max_players, 7) int64
        """
        batch_size = grid_ids.shape[0]
        grid_flat = grid_ids.view(batch_size, -1)
        grid_tokens = self.grid_id_embed(grid_flat)

        x_ids = self.grid_x_idx.unsqueeze(0).expand(batch_size, -1)
        y_ids = self.grid_y_idx.unsqueeze(0).expand(batch_size, -1)
        grid_tokens = (
            grid_tokens
            + self.x_embed(x_ids)
            + self.y_embed(y_ids)
            + self.token_type_embed(torch.zeros_like(x_ids))
        )

        player_x = player_features[:, :, 0]
        player_y = player_features[:, :, 1]
        player_has_flag = player_features[:, :, 2]
        player_in_prison = player_features[:, :, 3]
        player_team = player_features[:, :, 4]
        player_index = player_features[:, :, 5]
        player_valid = player_features[:, :, 6]

        player_tokens = (
            self.x_embed(player_x)
            + self.y_embed(player_y)
            + self.flag_embed(player_has_flag)
            + self.prison_embed(player_in_prison)
            + self.team_embed(player_team)
            + self.player_index_embed(player_index)
            + self.valid_embed(player_valid)
            + self.token_type_embed(torch.ones_like(player_x))
        )

        tokens = torch.cat([grid_tokens, player_tokens], dim=1)
        tokens = self.pre_norm(tokens)
        tokens = self.transformer(tokens)

        grid_token_count = self.width * self.height
        my_player_tokens = tokens[:, grid_token_count:grid_token_count + self.max_players]
        logits = self.action_head(my_player_tokens)

        pooled = tokens.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value
