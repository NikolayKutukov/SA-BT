"""SurvTRACE model classes (vendored + cleaned up).

Adapted from https://github.com/RyanWangZf/SurvTRACE (MIT License).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .modeling_bert import BaseModel, BertEmbeddings, BertEncoder, BertCLS, BertCLSMulti
from .utils import pad_col


class SurvTraceSingle(BaseModel):
    """SurvTRACE for single-event survival analysis."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertCLS(config)
        self.config = config
        self.init_weights()
        self._duration_index = config.get("duration_index")
        self.use_gpu = False

    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def forward(
        self,
        input_ids=None,
        input_nums=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions or self.config.get("output_attentions", False)
        output_hidden_states = output_hidden_states or self.config.get("output_hidden_states", False)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        head_mask = self.get_head_mask(head_mask, self.config["num_hidden_layers"])

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_x_num=input_nums,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[1]
        predict_logits = self.cls(encoder_outputs[0])

        return sequence_output, predict_logits

    def predict(self, x_input, batch_size=None):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input.iloc[:, : self.config["num_categorical_feature"]]
            x_input_num = x_input.iloc[:, self.config["num_categorical_feature"] :]
            x_num = torch.tensor(x_input_num.values).float()
            x_cat = torch.tensor(x_input_cat.values).long()
        else:
            x_cat = x_input[:, : self.config["num_categorical_feature"]].long()
            x_num = x_input[:, self.config["num_categorical_feature"] :].float()

        if self.use_gpu:
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()

        self.eval()
        with torch.no_grad():
            if batch_size is None:
                preds = self.forward(x_cat, x_num)[1]
            else:
                preds = []
                num_batch = int(np.ceil(len(x_num) / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx * batch_size : (idx + 1) * batch_size]
                    batch_x_cat = x_cat[idx * batch_size : (idx + 1) * batch_size]
                    batch_pred = self.forward(batch_x_cat, batch_x_num)
                    preds.append(batch_pred[1])
                preds = torch.cat(preds)
        return preds

    def predict_hazard(self, input_ids, batch_size=None):
        preds = self.predict(input_ids, batch_size)
        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard

    def predict_risk(self, input_ids, batch_size=None):
        surv = self.predict_surv(input_ids, batch_size)
        return 1 - surv

    def predict_surv(self, input_ids, batch_size=None, epsilon=1e-7):
        hazard = self.predict_hazard(input_ids, batch_size)
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv

    def predict_surv_df(self, input_ids, batch_size=None):
        surv = self.predict_surv(input_ids, batch_size)
        return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)


class SurvTraceMulti(BaseModel):
    """SurvTRACE for competing-events survival analysis."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertCLSMulti(config)
        self.config = config
        self.init_weights()
        self._duration_index = config.get("duration_index")
        self.use_gpu = False

    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def forward(
        self,
        input_ids=None,
        input_nums=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        event=0,
    ):
        output_attentions = output_attentions or self.config.get("output_attentions", False)
        output_hidden_states = output_hidden_states or self.config.get("output_hidden_states", False)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        head_mask = self.get_head_mask(head_mask, self.config["num_hidden_layers"])

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_x_num=input_nums,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[0]
        predict_logits = self.cls(sequence_output, event=event)
        return sequence_output, predict_logits

    def predict(self, x_input, batch_size=None, event=0):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input.iloc[:, : self.config["num_categorical_feature"]]
            x_input_num = x_input.iloc[:, self.config["num_categorical_feature"] :]
            x_num = torch.tensor(x_input_num.values).float()
            x_cat = torch.tensor(x_input_cat.values).long()
        else:
            x_cat = x_input[:, : self.config["num_categorical_feature"]].long()
            x_num = x_input[:, self.config["num_categorical_feature"] :].float()

        if self.use_gpu:
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()

        self.eval()
        with torch.no_grad():
            if batch_size is None:
                preds = self.forward(x_cat, x_num, event=event)[1]
            else:
                preds = []
                num_batch = int(np.ceil(len(x_num) / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx * batch_size : (idx + 1) * batch_size]
                    batch_x_cat = x_cat[idx * batch_size : (idx + 1) * batch_size]
                    batch_pred = self.forward(batch_x_cat, batch_x_num, event=event)
                    preds.append(batch_pred[1])
                preds = torch.cat(preds)
        return preds

    def predict_hazard(self, input_ids, batch_size=None, event=0):
        preds = self.predict(input_ids, batch_size, event=event)
        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard

    def predict_risk(self, input_ids, batch_size=None, event=0):
        surv = self.predict_surv(input_ids, batch_size, event=event)
        return 1 - surv

    def predict_surv(self, input_ids, batch_size=None, event=0):
        hazard = self.predict_hazard(input_ids, batch_size, event=event)
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv

    def predict_surv_df(self, input_ids, batch_size=None, event=0):
        surv = self.predict_surv(input_ids, batch_size, event=event)
        return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)
