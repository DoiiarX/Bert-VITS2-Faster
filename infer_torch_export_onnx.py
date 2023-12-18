import os
import logging

import re_matching


logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer_ import infer, latest_version, get_net_g

import gradio as gr
import numpy as np
from config import config
from scipy.io.wavfile import write


net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
):
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []
    if not cut_by_sent:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio = infer(
                p,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
    else:
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
                audio = infer(
                    s,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    skip_start=skip_start,
                    skip_end=skip_end,
                    g_model_name=g_model_name
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
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    g_model_name = config.webui_config.model.split('/')[-1].split('.')[0]
    g_model_name = None  # vits2转onnx就注释这一行，原始torch推理，则打开
    print('g model name', g_model_name)
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH","EN", "mix", "auto"]
    
    # 文本生成

    texts = ["你在玩什么游戏？你在玩什么游戏？你在玩什么游戏？你在玩什么游戏？",  "你在玩什么游戏？", "没有比你更美的山川。没有比你更纯的心田。漫过无边的黄沙白雪。是你陪我走过的日子。", "再来一句吧！再来两句。", "你好。","随便写一些新的句子，能快速吗？","依托答辩！", "新的句子，新的气象。","一个女人和一个女孩走在乡村路上。"]
    texts.append("我现在随便怎么写，也不好增加编译时间吧？")
    texts.append("某些特定的操作可能不支持动态维度。如果可能，尝试修改模型以避开这些操作。这可能涉及到重新设计模型的某些部分，或者替换一些层。")
    texts.append("保持数据在内存中的连续性是重要的。某些操作要求操作的张量在内存中是连续的。当一个张量在内存中连续时，它的元素是按照一定顺序线性排列的，没有间隔。")
    speaker = speakers[0]
    sdp_ratio = 0.2  # 0-1 # SDP/DP混合比
    noise_scale = 0.6  # 0.1-2  # 感情
    noise_scale_w = 0.8 # 0.1-2  # 音素长度
    length_scale = 1.0 # 0.1-2  # 语速
    language = languages[0]
    # 切分生成
    interval_between_sent = 0.2  # 0-5, 句间停顿(秒)
    interval_between_para = 1.0  # 0-10, 段间停顿(秒)
    opt_cut_by_sent = True  # 是否按句切分

    import time
    costs = dict()
    for i, text in enumerate(texts):
        start = time.time()
        text_output, audio_output = tts_split(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, \
            opt_cut_by_sent,interval_between_para,interval_between_sent)
        end = time.time()
        costs[i] = end - start
        samplerate = audio_output[0]
        audio_data = audio_output[1]
        write(f"example_xx_new_torch.wav", samplerate, audio_data.astype(np.int16))
    print(costs)

