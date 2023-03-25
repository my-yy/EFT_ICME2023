import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.position_encode import PositionalEncoding


class Model(nn.Module):

    def __init__(self,
                 features_names, input_dim_trans, num_head, dim_feedforward, nlayers, dropout,
                 modal_token_std,
                 use_final_project_layer,
                 use_pos_emb
                 ):

        super(Model, self).__init__()
        self.input_dim_trans = input_dim_trans
        self.use_final_project_layer = use_final_project_layer
        encoder_layers = TransformerEncoderLayer(input_dim_trans, num_head, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # type_token
        self.use_modal_token = modal_token_std > 0
        if self.use_modal_token:
            def init_weights(module):
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=modal_token_std)

            self.type_embedding = torch.nn.Embedding(2, input_dim_trans)
            self.type_embedding.apply(init_weights)

        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_encoder = PositionalEncoding(input_dim_trans)
            pos_ratio = 1e-5
            self.pos_ratio = pos_ratio

        voice_features = [f for f in features_names if f.startswith("v")]
        face_features = [f for f in features_names if f.startswith("f")]
        self.face_features_count = len(face_features)

        self.create_project_layers("v", voice_features)
        self.create_project_layers("f", face_features)

        embedding_size = 512
        self.face_project_layer = self.get_final_project_layer(self.input_dim_trans, embedding_size)
        self.voice_project_layer = self.get_final_project_layer(self.input_dim_trans, embedding_size)

    def get_final_project_layer(self, input_size, output_size):
        if self.use_final_project_layer:
            return torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size)
            )
        else:
            return torch.nn.Identity()

    def create_project_layers(self, name, voice_features):
        for i in range(len(voice_features)):
            featue_name = voice_features[i]
            featue_raw_dim = int(featue_name.split("_")[-1])
            k = "project_" + name + str(i)

            v = self.get_pre_fc(featue_raw_dim, self.input_dim_trans)
            setattr(self, k, v)

    def get_pre_fc(self, in_size, out_size):
        return torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(in_size, out_size)
        )

    def forward(self, input_list):
        face_count = self.face_features_count

        face = input_list[0:face_count]
        voice = input_list[face_count:]

        voice_emb = self.voice_encoder(voice)
        face_emb = self.face_encoder(face)

        return voice_emb, face_emb

    def face_encoder(self, input_list):
        emb = self.common_encoder(input_list, "f")
        return self.face_project_layer(emb)

    def voice_encoder(self, input_list):
        emb = self.common_encoder(input_list, "v")
        return self.voice_project_layer(emb)

    def common_encoder(self, input_list, name):
        assert name in ["v", "f"]
        the_features = []
        for i in range(len(input_list)):
            k = "project_" + name + str(i)
            layer = getattr(self, k)
            feat = layer(input_list[i])
            the_features.append(feat)

        feature = torch.stack(the_features)

        # modal token
        if self.use_modal_token:
            idx = ["v", "f"].index(name)
            batch_size = input_list[0].shape[0]
            emb = self.type_embedding(torch.LongTensor([[idx] * batch_size]).cuda())
            # [1,batch,512]
            feature = torch.cat([emb, feature], dim=0)

        return self.calc_emb(feature)

    def calc_emb(self, input_val):
        if self.use_pos_emb:
            input_val = self.pos_encoder(input_val, self.pos_ratio)

        encoder_result = self.transformer_encoder(input_val, src_key_padding_mask=None)
        # [seq_len, batch, feature_dim]

        output = torch.mean(encoder_result, dim=0)
        # [batch, feature_dim]
        return output
