import os,re,sys
import LangSegment
import gradio as gr
import librosa,pdb
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert
from time import time as ttime
from datetime import datetime
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.mel_processing import spectrogram_torch
from module.models import SynthesizerTrn
from my_utils import load_audio
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
import pytz
import soundfile as sf
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
tz = pytz.timezone('Asia/Singapore')
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "openai/whisper-medium"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

def abs_path(dir):
    global_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    return(os.path.join(global_dir, dir))
gpt_path = abs_path("MODELS/33/33.ckpt")
sovits_path=abs_path("MODELS/33/33.pth")

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "pretrained_models/chinese-roberta-wwm-ext-large"
)

from timeit import default_timer as timer
#cnhubert_base_path =  "TencentGameMate/chinese-hubert-base"
#bert_path =  "hfl/chinese-roberta-wwm-ext-large"
cnhubert.cnhubert_base_path = cnhubert_base_path
is_half = eval(
    os.environ.get("is_half", "True" if torch.cuda.is_available() else "False")
)
device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)


change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {
    ("中文1"): "all_zh",#全部按中文识别
    ("English"): "en",#全部按英文识别#######不变
    ("日文1"): "all_ja",#全部按日文识别
    ("中文"): "zh",#按中英混合识别####不变
    ("日本語"): "ja",#按日英混合识别####不变
    ("混合"): "auto",#多语种启动切分识别语种
}


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist


def clean_text_inf(text, language):
    formattext = ""
    language = language.replace("all_","")
    for tmp in LangSegment.getTexts(text):
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_cleaned_text_final(text,language):
    if language in {"en","all_zh","all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja","auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text

def get_bert_final(phones, word2ph, text,language,device):
    if language == "en":
        bert = get_bert_inf(phones, word2ph, text, language)
    elif language in {"zh", "ja","auto"}:
        bert = nonen_get_bert_inf(text, language)
    elif language == "all_zh":
        bert = get_bert_feature(text, word2ph).to(device)
    else:
        bert = torch.zeros((1024, len(phones))).to(device)
    return bert

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def tprint(text):
    now=datetime.now(tz).strftime('%H:%M:%S')
    print(f'UTC+8 - {now} - ✅{text}')

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=("Do not split"),playback_speed=1.0, volume_scale=1.0):
    t0 = ttime()
    startTime=timer()
    change_sovits_weights(sovits_path)
    tprint(f'LOADED SoVITS Model: {sovits_path}')
    change_gpt_weights(gpt_path)
    tprint(f'LOADED GPT Model: {gpt_path}')

    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    prompt_text = prompt_text.strip("\n")
    if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
    print(("实际输入的参考文本:"), prompt_text)
    print(("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            raise OSError(("参考音频在3~10秒范围外，请更换！"))
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()

    phones1, word2ph1, norm_text1=get_cleaned_text_final(prompt_text, prompt_language)

    if (how_to_cut == ("Split into groups of 4 sentences")):
        text = cut1(text)
    elif (how_to_cut == ("Split every 50 characters")):
        text = cut2(text)
    elif (how_to_cut == ("Split at CN/JP periods (。)")):
        text = cut3(text)
    elif (how_to_cut == ("Split at English periods (.)")):
        text = cut4(text)
    elif (how_to_cut == ("Split at punctuation marks")):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    bert1=get_bert_final(phones1, word2ph1, norm_text1,prompt_language,device).to(dtype)

    for text in texts:
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(("实际输入的目标文本(每句):"), text)
        phones2, word2ph2, norm_text2 = get_cleaned_text_final(text, text_language)
        bert2 = get_bert_final(phones2, word2ph2, norm_text2, text_language, device).to(dtype)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config["inference"]["top_k"],
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
                .detach()
                .cpu()
                .numpy()[0, 0]
        ) 
        max_audio=np.abs(audio).max()
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    #yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    audio_data = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    if playback_speed != 1.0:
        audio_data_float = audio_data.astype(np.float32) / 32768  
        audio_data_stretched = librosa.effects.time_stretch(audio_data_float, rate=playback_speed)
        audio_data = (audio_data_stretched * 32768).astype(np.int16)  
    audio_data = (audio_data.astype(np.float32) * volume_scale).astype(np.int16)
    output_wav = "output_audio.wav"  
    sf.write(output_wav, audio_data, hps.data.sampling_rate)
    endTime=timer()
    tprint(f'TTS COMPLETE,{round(endTime-startTime,4)}s')
    return output_wav

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：]'
    items = re.split(f'({punds})', inp)
    items = ["".join(group) for group in zip(items[::2], items[1::2])]
    opt = "\n".join(items)
    return opt


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def update_model(choice="女神"):
    global gpt_path, sovits_path  
    model_info = models[choice]
    gpt_path = abs_path(model_info["gpt_weight"])
    sovits_path = abs_path(model_info["sovits_weight"])
    model_name = choice
    tone_info = model_info["tones"]["tone1"] 
    tprint(f'SELECT MODEL：{choice}')
    # 返回默认tone“tone1”
    return (
        tone_info["example_voice_wav"],   
        tone_info["example_voice_wav_words"],   
        model_info["default_language"],   
        model_info["default_language"],
        model_name,
        "tone1"  
    )

def update_tone(model_choice, tone_choice):
    model_info = models[model_choice]  
    tone_info = model_info["tones"][tone_choice]  
    example_voice_wav = abs_path(tone_info["example_voice_wav"])  
    example_voice_wav_words = tone_info["example_voice_wav_words"]  
    return example_voice_wav, example_voice_wav_words

def transcribe(voice):
    time1=timer()
    tprint('Start transcribe')
    task="transcribe"
    if voice is None:
        print("No audio file submitted! Please upload or record an audio file before submitting your request.")
    R = pipe(voice, batch_size=8, generate_kwargs={"task": task}, return_timestamps=True,return_language=True)
    text=R['text']
    lang=R['chunks'][0]['language']
    if lang=='english':
      language='English'
    elif lang =='chinese':
      language='中文'
    elif lang=='japanese':
      language = '日本語'

    time2=timer()
    tprint(f'TRANSCRIBE COMPLETE,{round(time2-time1,4)}s')
    print(f'language：{language}，words：{text}')
    return  text,language  

def clone_voice(user_voice,user_text,user_lang):
    tprint('Start clone')
    time1=timer()
    global gpt_path, sovits_path
    gpt_path = abs_path("pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
    #tprint(f'Model loaded:{gpt_path}')
    sovits_path = abs_path("pretrained_models/s2G488k.pth")
    #tprint(f'Model loaded:{sovits_path}')
    prompt_text, prompt_language = transcribe(user_voice)
    output_wav = get_tts_wav(
    user_voice,
    prompt_text,
    prompt_language,
    user_text,
    user_lang,
    how_to_cut="Do not split",
    playback_speed=1.0,
    volume_scale=1.0)
    time2=timer()
    tprint(f'CLONE COMPLETE,{round(time2-time1,4)}s')
    return output_wav


from info import models
models_by_language = {
    "English": [],
    "中文": [],
    "日本語": []
}
for model_name, model_info in models.items():
    language = model_info["default_language"]
    models_by_language[language].append((model_name, model_info))

##########GRADIO###########

with gr.Blocks(theme='remilia/Ghostly') as app:
    gr.HTML('''
  <h1 style="font-size: 25px;">A TTS GENERATOR</h1>
  <p style="margin-bottom: 10px; font-size: 100%">
    This space is based on the innovative text-to-speech generation solution
    <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">GPT-SoVITS</a> .
    You can visit the repo's github homepage to learn training and inference.<br>
    本空间基于新式的文字转语音生成方案 <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">GPT-SoVITS</a> .
    你可以前往项目的github主页学习如何推理和训练。<br>
    Due to using Hugging Face's free CPU for inference in this space, the speed of generating voice 
    is very slow. If you want to generate voice more quickly, please click the Colab icon below to go to Colab 
    and use this space, which will greatly improve the generation speed.<br>
    由于本空间使用huggingface的免费CPU进行推理，因此速度很慢，如果你想获得快速的推理，
    请点击下方的Colab图标，前往Colab使用本空间，会大大提升生成语音的速度
  </p>
   <a href="https://colab.research.google.com/drive/1fTuPZ4tZsAjS-TrhQWMCb7KRdnU8aF6j#scrollTo=MDtJIbLdLHe9" target="_blank"><img src="https://camo.githubusercontent.com/dd83d4a334eab7ada034c13747d9e2237182826d32e3fda6629740b6e02f18d8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f6c61622d4639414230303f7374796c653d666f722d7468652d6261646765266c6f676f3d676f6f676c65636f6c616226636f6c6f723d353235323532" alt="aolab"></a>
''')

    default_voice_wav, default_voice_wav_words, default_language, _, default_model_name, _ = update_model("Trump")
    english_models = [name for name, _ in models_by_language["English"]]
    chinese_models = [name for name, _ in models_by_language["中文"]]
    japanese_models = [name for name, _ in models_by_language["日本語"]]
    with gr.Row():
        english_choice = gr.Radio(english_models, label="EN|English Model",value="Trump")
        chinese_choice = gr.Radio(chinese_models, label="CN|中文模型")
        japanese_choice = gr.Radio(japanese_models, label="JP|日本語モデル")

    plsh='Text must match the selected language option to prevent errors, for example, if English is input but Chinese is selected for generation./文字一定要和语言选项匹配，不然要报错，比如输入的是英文，生成语言选中文'
    with gr.Row():
        model_name = gr.Textbox(label="Seleted Model/已选模型", value=default_model_name, scale=1) 
        text = gr.Textbox(label="Input some text for voice generation/输入想要生成语音的文字", lines=5,scale=8,
        placeholder=plsh)


    with gr.Row():
        text_language = gr.Radio(
            label="Select language for input text/输入的文字对应语言",
            choices=["中文","English","日本語"],
            value=default_language,
            info='Input text and language must match.',scale=1,
            )
        tone_select = gr.Radio(
            label="Select Tone/选择语气",
            choices=["tone1","tone2","tone3"],
            value="tone1",
            info='Tone influences the emotional expression ',scale=1)
        
        how_to_cut = gr.Dropdown(
                label=("How to split?"),
                choices=[("Do not split"), ("Split into groups of 4 sentences"), ("Split every 50 characters"), 
                         ("Split at CN/JP periods (。)"), ("Split at English periods (.)"), ("Split at punctuation marks"), ],
                value=("Split into groups of 4 sentences"),
                interactive=True,
            info='A suitable splitting method can achieve better generation results',scale=2
            )
    with gr.Accordion(label="Preview selected tone/预览语气", open=False):
        with gr.Row(visible=True):
            inp_ref = gr.Audio(label="Reference audio", type="filepath", value=default_voice_wav, scale=3)
            prompt_text = gr.Textbox(label="Reference text", value=default_voice_wav_words, scale=3)
            prompt_language = gr.Dropdown(label="Language of the reference audio", choices=["中文", "English", "日本語"], value=default_language, scale=1,interactive=False)

    tone_select.change(update_tone, inputs=[model_name, tone_select], outputs=[inp_ref, prompt_text])
    
    with gr.Accordion(label="Additional generation options/附加生成选项", open=False):
        volume = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.01, label='Volume')
        speed = gr.Slider(minimum=0.5, maximum=1.5, value=1, step=0.05, label='Speed')
    
    
    with gr.Row():
        main_button = gr.Button("✨Generate Voice", variant="primary", scale=1)
        output = gr.Audio(label="💾Download it by clicking ⬇️", scale=3)
        #info = gr.Textbox(label="INFO", visible=True, readonly=True, scale=1)

    gr.HTML('''<br><br>
    <h1 style="font-size: 25px;">Clone custom Voice/克隆自定义声音</h1>
    <p style="margin-bottom: 10px; font-size: 100%">Need 3~10s audio.This involves voice-to-text conversion followed by text-to-voice conversion, so it takes longer time<br>
    需要3~10秒语音，这个会涉及语音转文字，之后再转语音，所以耗时比较久
    </p>''')
    with gr.Row():
        user_voice = gr.Audio(sources=["microphone", "upload"],type="filepath", label="（3~10s）Upload or Record audio/上传或录制声音",scale=3)
        user_text= gr.Textbox(label="Text for generation/输入想要生成语音的文字", lines=5,scale=5,
        placeholder=plsh)
        user_lang = gr.Dropdown(label="Language/生成语言", choices=["中文", "English", "日本語"],scale=1)

    user_button = gr.Button("✨Clone Voice", variant="primary")
    user_output = gr.Audio(label="💾Output wave file,Download it by clicking ⬇️")
    english_choice.change(update_model, inputs=[english_choice], outputs=[inp_ref, prompt_text, prompt_language, text_language, model_name, tone_select])
    chinese_choice.change(update_model, inputs=[chinese_choice], outputs=[inp_ref, prompt_text, prompt_language, text_language, model_name, tone_select])
    japanese_choice.change(update_model, inputs=[japanese_choice], outputs=[inp_ref, prompt_text, prompt_language, text_language, model_name, tone_select])
    
    
    main_button.click(
    get_tts_wav,
    inputs=[inp_ref, prompt_text, prompt_language, text, text_language, how_to_cut,speed,volume],
    outputs=[output])

    user_button.click(
    clone_voice,
    inputs=[user_voice,user_text,user_lang],
    outputs=[user_output])

app.launch(share=True)