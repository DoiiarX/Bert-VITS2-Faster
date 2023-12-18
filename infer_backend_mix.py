import time
import os
import logging
from typing import Any
from collections import defaultdict

import re_matching

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import utils
import gradio as gr
import numpy as np
from config import config
from scipy.io.wavfile import write

latest_version = "2.0"
device = config.webui_config.device

from text import cleaned_text_to_sequence
from text.cleaner import clean_text
import commons

import torch
import numpy as np

from polygraphy.backend.trt import(
    EngineFromBytes, TrtRunner
)
from polygraphy.logger import G_LOGGER

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

from transformers import AutoTokenizer

# -------------------------BERT TRT---------------------------------------
class TRTModelInfer():
    def __init__(self, engine_path) -> None:
        self.engine_path = engine_path
        with open(engine_path, 'rb') as f:
            self.engine = EngineFromBytes(f.read())
            
        with G_LOGGER.verbosity(G_LOGGER.CRITICAL):
            self.runner = TrtRunner(self.engine)
            self.runner.__enter__()
            
    def __call__(self, inputs_dict, **kwds: Any) -> Any:
        outputs = self.runner .infer(inputs_dict)
        return outputs
    
    def __del__(self):
        self.runner.__exit__(None, None, None)

def get_bert_feature_trt(text, word2ph):
    token_time = time.time()
    inputs = tokenizer(text, return_tensors="np")
    token_cost = time.time() - token_time
    print('cost of token ', token_cost)
    
    roberta_time = time.time()
    res = TRTI(inputs)["last_hidden_state"]
    roberta_cost = time.time() - roberta_time
    print('roberta_cost time',roberta_cost)
    
    res = torch.from_numpy(res)[0].cpu()

    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

def get_bert_trt(norm_text, word2ph, language):
    lang_bert_func_map = {"ZH": get_bert_feature_trt}
    bert = lang_bert_func_map[language](norm_text, word2ph)
    return bert

# -------------------------BERT TRT---------------------------------------

# -------------------------MIX ALL---------------------------------------
class SynthesizerTrnMixed():
    def __init__(self, onnx_path_dir) -> None:
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.onnx_path_dir = onnx_path_dir
        self.sub_model_num = defaultdict(int)
        self.providers_conig = [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 
                                                                    'trt_engine_cache_path': './cache_onnx_ep'}), 'CUDAExecutionProvider']
        
        for p in os.listdir(self.onnx_path_dir):
            onnx_path = os.path.join(self.onnx_path_dir, p)
            if 'sdp' in p:
                self.sdp = TRT_SDP
                self.sub_model_num["sdp_sim"] = 1
            elif 'dec' in p:
                # self.dec = onnxruntime.InferenceSession(onnx_path, sess_options, providers=self.providers_conig)
                self.dec = TRT_DEC
                self.sub_model_num["dec_sim"] = 1
            elif 'dp' in p:
                # self.dp = onnxruntime.InferenceSession(onnx_path, sess_options, providers=self.providers_conig)
                self.dp = TRT_DP
                self.sub_model_num['dp_sim'] = 1
            elif 'emb' in p:
                self.emb = onnxruntime.InferenceSession(onnx_path, sess_options, providers=self.providers_conig)
                self.sub_model_num['emb_sim'] = 1
            elif 'flow' in p:
                # self.flow = onnxruntime.InferenceSession(onnx_path, sess_options, providers=self.providers_conig)
                self.flow = TRT_FLOW
                self.sub_model_num['flow_sim'] = 1
            elif 'enc' in p:
                self.enc = TRT_ENC
                self.sub_model_num['enc_sim'] = 1
            else:
                pass
            if len(self.sub_model_num) == 6:
                break
        if len(self.sub_model_num) != 6:
            print(self.sub_model_num)
            raise ValueError('smoe model loda failed')
        else:
            print('all submmodel loaded!')
        
    def get_text_trt(self, text, language_str, hps):
        norm_text, phone, tone, word2ph = clean_text(text, language_str)
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

        if hps.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_ori = get_bert_trt(norm_text, word2ph, language_str)
        
        
        
        del word2ph
        assert bert_ori.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "EN":
            # 省去了
            bert = bert_ori
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        else:
            raise ValueError("language_str should be ZH, EN")

        assert bert.shape[-1] == len(
            phone
        ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, en_bert, phone, tone, language
    
    def __call__(self,
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                language,
                hps) -> Any:
        
        tt = time.time()
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.get_text_trt(text, language, hps)
        cost_text = time.time() - tt
        print('get text trt cost', cost_text)

        x_tst = phones.numpy()[np.newaxis, :]
        x_tst_lengths = np.array([phones.size(0)], dtype=np.int64)
        speakers = np.array([hps.data.spk2id[sid]], dtype=np.int64)
        tones = tones.numpy()[np.newaxis, :]
        lang_ids = lang_ids.numpy()[np.newaxis, :]
        bert = bert.numpy()[np.newaxis, :]
        ja_bert = ja_bert.numpy()[np.newaxis, :]
        en_bert = en_bert.numpy()[np.newaxis, :]
        
        inputs_emb = {'sid':speakers}
        t0 = time.time()
        g = self.emb.run(None, inputs_emb)[0][..., np.newaxis]
        cost_emb = time.time() - t0
        print('cost_emb', cost_emb)
        inputs_enc = {
            "x":x_tst,
            "x_lengths":x_tst_lengths,
            "tone": tones,
            "language": lang_ids,
            "bert": bert,
            "ja_bert": ja_bert,
            "en_bert": en_bert,
            "g": g
        }

        t1 = time.time()
        x, m_p, logs_p, x_mask = list(self.enc(inputs_enc).values())
        cost_enc = time.time() - t1
        print('cost_enc', cost_enc)

        inputs_sdp = {
            "x":x,
            "x_mask":x_mask,
            "noise_scale_w": noise_scale_w*np.ones((1,), dtype=np.float32),
            "g":g
        }

        inputs_dp = {
            "x":x,
            "x_mask":x_mask,
            "g":g
        }
        t2 = time.time()
        logw = self.sdp(inputs_sdp)['logw'] * sdp_ratio + self.dp(inputs_dp)['logw'] * (1 - sdp_ratio)
        cost_sdp_dp = time.time() - t2
        print('cost_sdp_dp', cost_sdp_dp)

        # 比较多的操作，直接转为tensor运算
        t3 = time.time()
        logw = torch.from_numpy(logw).to('cuda')
        x_mask = torch.from_numpy(x_mask).to('cuda')
        m_p = torch.from_numpy(m_p).to('cuda')
        logs_p = torch.from_numpy(logs_p).to('cuda')
        
        w = torch.exp(logw) * x_mask * length_scale
        
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        
        z_p = z_p.cpu().numpy()
        y_mask = y_mask.cpu().numpy()
        cost_np_tensor = time.time() - t3
        print('cost_np_tensor', cost_np_tensor)
        inputs_flow = {
            "z_p":z_p,
            "y_mask": y_mask,
            "g":g
        }
        t4 = time.time()
        # z = self.flow.run(None, inputs_flow)[0]  # 0.005
        z = TRT_FLOW(inputs_flow)['z']  # 0.017
        cost_flow = time.time() - t4
        print('cost_flow', cost_flow)
        z_in = z * y_mask
        inputs_dec = {
            'z_in':z_in,
            "g":g
        }
        t5 = time.time()
        o = self.dec(inputs_dec)['o']
        
        
        cost_dec = time.time() - t5
        print('cost_dec', cost_dec)
        vits2_cost = cost_dec + cost_emb + cost_enc + cost_flow + cost_np_tensor + cost_sdp_dp
        print('all cost is:', vits2_cost + cost_text, "vits2_cost is:", vits2_cost)
        return o[0, 0]
# -------------------------MIX ALL---------------------------------------

# -------------------------INFERE BY SENTENCE---------------------------------------
def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    interval_between_para,
    interval_between_sent,
):
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []

    print('cut_by_sent')
    for idx, p in enumerate(para_list):
        skip_start = idx != 0
        skip_end = idx != len(para_list) - 1
        audio_list_sent = []
        sent_list = re_matching.cut_sent(p)
        for idx, s in enumerate(sent_list):
            print(idx, 'idx')
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(sent_list) - 1) and skip_end
            audio = SSO(
                s,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps
            )
            audio_list_sent.append(audio)
            silence = np.zeros((int)(44100 * interval_between_sent))
            audio_list_sent.append(silence)
        if (interval_between_para - interval_between_sent) > 0:
            silence = np.zeros(
                (int)(44100 * (interval_between_para - interval_between_sent))
            )
            audio_list_sent.append(silence)
        audio16bit = gr.processing_utils.convert_to_16_bit_wav(
            np.concatenate(audio_list_sent)
        )  # 对完整句子做音量归一
        audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))


if __name__ == "__main__":
    engine_path = "./engines/chinese_roberta_wwm_ext_large_model_fp16_dynamic.engine"
    TRTI = TRTModelInfer(engine_path)
    engine_path_enc = "./engines/_enc_.engine"
    TRT_ENC = TRTModelInfer(engine_path_enc)
    engine_path_sdp = "./engines/_sdp_.engine"
    TRT_SDP = TRTModelInfer(engine_path_sdp)
    engine_path_flow = "./engines/_flow_new.engine"
    TRT_FLOW = TRTModelInfer(engine_path_flow)
    engine_path_dp = "./engines/_dp_.engine"
    TRT_DP = TRTModelInfer(engine_path_dp)
    engine_path_dec = "./engines/_dec_.engine"
    TRT_DEC = TRTModelInfer(engine_path_dec)
    LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    version = hps.version if hasattr(hps, "version") else latest_version
    
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "EN", "mix", "auto"]
    
    # 文本生成
    # texts = ['某些特定的操作可能不支持动态维度。如果可能，尝试修改模型以避开这些操作。这可能涉及到重新设计模型的某些部分，或者替换一些层']
    texts = ["你在玩什么游戏？你在玩什么游戏？你在玩什么游戏？你在玩什么游戏？",  "你在玩什么游戏？", "没有比你更美的山川。没有比你更纯的心田。漫过无边的黄沙白雪。是你陪我走过的日子。", "再来一句吧！再来两句。", "你好。","随便写一些新的句子，能快速吗？","依托答辩！", "新的句子，新的气象。","一个女人和一个女孩走在乡村路上。"]
    texts.append("某些特定的操作可能不支持动态维度。如果可能，尝试修改模型以避开这些操作。这可能涉及到重新设计模型的某些部分，或者替换一些层。")

    speaker = speakers[0]
    sdp_ratio = 0.2  # 0-1 # SDP/DP混合比
    noise_scale = 0.6  # 0.1-2  # 感情
    noise_scale_w = 0.8 # 0.1-2  # 音素长度
    length_scale = 1.0 # 0.1-2  # 语速
    language = languages[0]
    
    # 切分生成
    interval_between_sent = 0.2  # 0-5, 句间停顿(秒)
    interval_between_para = 1.0  # 0-10, 段间停顿(秒)
    opt_cut_by_sent = True  # 是否按句切分 默认
    
    SSO = SynthesizerTrnMixed('./onnx_exports/G_122000')
    import time
    costs = dict()
    for i, text in enumerate(texts):
        start = time.time()
        text_output, audio_output = tts_split(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, \
            interval_between_para,interval_between_sent)
        end = time.time()
        costs[i] = end - start
        samplerate = audio_output[0]
        audio_data = audio_output[1]
        write(f"outs/trt_onnx_torch_bert_vits2_{i}_mixed.wav", samplerate, audio_data.astype(np.int16))
    print(costs)
