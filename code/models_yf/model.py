import tensorflow as tf
from .embed import DataEmbedding
from .attn import ProbAttention, FullAttention, AttentionLayer
from .encoder import ConvLayer, Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer


class Informer(tf.keras.Model):
    def __init__(self, args):
        super(Informer, self).__init__()
        self.args = args
        # Encoding
        self.enc_embedding = DataEmbedding(self.args.enc_in, self.args.d_model, self.args.embed, self.args.dropout)
        self.dec_embedding = DataEmbedding(self.args.dec_in, self.args.d_model, self.args.embed, self.args.dropout)
        # Attention
        Attn = ProbAttention if self.args.attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, self.args.factor, attention_dropout=self.args.dropout),
                                   self.args.d_model, self.args.n_heads),
                    self.args.d_model,
                    self.args.d_ff,
                    dropout=self.args.dropout,
                    activation=self.args.activation
                ) for l in range(self.args.e_layers)
            ],
            [
                ConvLayer(
                    self.args.d_model
                ) for l in range(self.args.e_layers - 1)
            ],
            norm_layer=tf.keras.layers.LayerNormalization()
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, self.args.factor, attention_dropout=self.args.dropout),
                                   self.args.d_model, self.args.n_heads),
                    AttentionLayer(FullAttention(False, self.args.factor, attention_dropout=self.args.dropout),
                                   self.args.d_model, self.args.n_heads),
                    self.args.d_model,
                    self.args.d_ff,
                    dropout=self.args.dropout,
                    activation=self.args.activation,
                )
                for l in range(self.args.d_layers)
            ],
            norm_layer=tf.keras.layers.LayerNormalization()
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = tf.keras.layers.Dense(self.args.target_len)

    def call(self, inputs, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # if self.args.time_embedding:
        #     # embedding
        #     x_enc, x_dec, x_mark_enc, x_mark_dec = inputs
        #     x_mark_enc.set_shape((len(x_enc), self.args.seq_len, x_mark_enc.shape[2]))
        #     x_mark_dec.set_shape((len(x_dec), self.args.target_len, x_mark_dec.shape[2]))
        #     enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #     # x
        #     x_enc.set_shape((len(x_enc), self.args.seq_len, x_enc.shape[2]))
        #     x_dec.set_shape((len(x_dec), self.args.dec_input_len, x_dec.shape[2]))
        #     enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        #     dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # else:
        #     x_enc, x_dec, x_mark_enc, x_mark_dec = inputs
        #     enc_out = x_enc
        #     x_enc.set_shape((len(x_enc), self.args.seq_len, x_enc.shape[2]))
        #     x_dec.set_shape((len(x_dec), self.args.dec_input_len, x_dec.shape[2]))
        #     enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        #     dec_out = x_dec

        # embedding
        x_enc, x_dec, x_mark_enc, x_mark_dec = inputs
        x_mark_enc.set_shape((len(x_enc), self.args.seq_len, x_mark_enc.shape[2]))
        x_mark_dec.set_shape((len(x_dec), self.args.dec_input_len, x_mark_dec.shape[2]))
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # x
        x_enc.set_shape((len(x_enc), self.args.seq_len, x_enc.shape[2]))
        x_dec.set_shape((len(x_dec), self.args.dec_input_len, x_dec.shape[2]))
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)


        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.args.target_len:, :]  # [B, L, D]

