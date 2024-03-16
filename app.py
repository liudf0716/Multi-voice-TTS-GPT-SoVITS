import gradio as gr
import numpy as np
import soundfile as sf
from datetime import datetime
from time import time as ttime
from my_utils import load_audio
from transformers import pipeline
from text.cleaner import clean_text
from polyglot.detect import  Detector
from feature_extractor import cnhubert
from timeit import default_timer as timer
from text import cleaned_text_to_sequence
from module.models  import  SynthesizerTrn
from module.mel_processing import spectrogram_torch
from transformers.pipelines.audio_utils import ffmpeg_read
import os,re,sys,LangSegment,librosa,pdb,torch,pytz,random
from transformers import AutoModelForMaskedLM, AutoTokenizer
from AR.models.t2s_lightning_module import Text2SemanticLightningModule


import logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart").setLevel(logging.WARNING)
from download import *
download()
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
tz = pytz.timezone('Asia/Singapore')
device = "cuda" if torch.cuda.is_available() else "cpu"

def abs_path(dir):
    global_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    return(os.path.join(global_dir, dir))
gpt_path = abs_path("MODELS/22/22.ckpt")
sovits_path=abs_path("MODELS/22/22.pth")
cnhubert_base_path = os.environ.get("cnhubert_base_path", "pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "pretrained_models/chinese-roberta-wwm-ext-large")

if not os.path.exists(cnhubert_base_path):
    cnhubert_base_path = "TencentGameMate/chinese-hubert-base"
if not os.path.exists(bert_path):
    bert_path = "hfl/chinese-roberta-wwm-ext-large"
cnhubert.cnhubert_base_path = cnhubert_base_path

whisper_path = os.environ.get("whisper_path", "pretrained_models/whisper-tiny")
if not os.path.exists(whisper_path):
    whisper_path = "openai/whisper-tiny"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=whisper_path,
    chunk_length_s=30,
    device=device,)


is_half = eval(
    os.environ.get("is_half", "True" if torch.cuda.is_available() else "False")
)


dict_language = {
    "中文1": "all_zh",
    "English": "en",
    "日文1": "all_ja",
    "中文": "zh",
    "日本語": "ja",
    "混合": "auto",
}

cut_method = {
    "Do not split/不切":"cut0",
    "Split into groups of 4 sentences/四句一切": "cut1",
    "Split every 50 characters/50字一切": "cut2",
    "Split at CN/JP periods (。)/按中日文句号切": "cut3",
    "Split at English periods (.)/按英文句号切": "cut4",
    "Split at punctuation marks/按标点切": "cut5",
}


tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path

    
tts_pipline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path


def inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,
              volume
              ):

    if not duration(ref_audio_path):
        return None
    if  text == '':
        wprint("Please input text to generate/请输入生成文字")
        return None
    text=trim_text(text,text_language)             
    try:
        lang=dict_language[text_lang]
        inputs={
        "text": text,
        "text_lang": lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "volume":volume,
        "return_fragment":False,
        }
    
        yield next(tts_pipline.run(inputs))
    except KeyError as e:
        wprint(f'Unsupported language type:{e}')
        return None

#==========custom functions============

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
def tprint(text):
    now=datetime.now(tz).strftime('%H:%M:%S')
    print(f'UTC+8 - {now} - {text}')

def wprint(text):
    tprint(text)
    gr.Warning(text)

def lang_detector(text):
    min_chars = 5
    if len(text) < min_chars:
        return "Input text too short/输入文本太短"
    try:
        detector = Detector(text).language
        lang_info = str(detector)
        code = re.search(r"name: (\w+)", lang_info).group(1)
        if code == 'Japanese':
            return "日本語"
        elif code == 'Chinese':
            return "中文"
        elif code == 'English':
            return 'English'
        else:
            return code
    except Exception as e:
        return f"ERROR：{str(e)}"
        
def trim_text(text,language): 
    limit_cj = 120 #character
    limit_en = 60 #words  
    search_limit_cj = limit_cj+30
    search_limit_en = limit_en +30
    text = text.replace('\n', '').strip()
    
    if language =='English':
        words = text.split()
        if len(words) <= limit_en:
            return text
        # English
        for i in range(limit_en, -1, -1):
            if any(punct in words[i] for punct in splits):
                return ' '.join(words[:i+1])
        for i in range(limit_en, min(len(words), search_limit_en)):
            if any(punct in words[i] for punct in splits):
                return ' '.join(words[:i+1])
        return ' '.join(words[:limit_en])
        
    else:
        if len(text) <= limit_cj:
            return text
        for i in range(limit_cj, -1, -1):  
            if text[i] in splits:
                return text[:i+1]
        for i in range(limit_cj, min(len(text), search_limit_cj)):  
            if text[i] in splits:
                return text[:i+1]
        return text[:limit_cj]   

def duration(audio_file_path):
    if not audio_file_path:
        wprint("Failed to obtain uploaded audio/未找到音频文件")
        return False
    try:
        audio_duration = librosa.get_duration(filename=audio_file_path)
        if not 3 < audio_duration < 10:
            wprint("The audio length must be between 3~10 seconds/音频时长须在3~10秒之间")
            return False
        return True
    except FileNotFoundError:
        return False

def update_model(choice):
    #global tts_config.vits_weights_path, tts_config.t2s_weights_path
    model_info = models[choice]
    gpt_path = abs_path(model_info["gpt_weight"])
    sovits_path = abs_path(model_info["sovits_weight"])
    tts_pipline.init_vits_weights(sovits_path)
    tts_pipline.init_t2s_weights(gpt_path)
    model_name = choice
    tone_info = model_info["tones"]["tone1"] 
    tone_sample_path = abs_path(tone_info["sample"])
    tprint(f'✅SELECT MODEL：{choice}')
    # 返回默认tone“tone1”
    return (
        tone_info["example_voice_wav"],   
        tone_info["example_voice_wav_words"],   
        model_info["default_language"],   
        model_info["default_language"],
        model_name,
        "tone1"  ,
        tone_sample_path
    )

def update_tone(model_choice, tone_choice):
    model_info = models[model_choice]  
    tone_info = model_info["tones"][tone_choice]  
    example_voice_wav = abs_path(tone_info["example_voice_wav"])  
    example_voice_wav_words = tone_info["example_voice_wav_words"]  
    tone_sample_path = abs_path(tone_info["sample"])
    return example_voice_wav, example_voice_wav_words,tone_sample_path

def transcribe(voice):
    time1=timer()
    tprint('⚡Start Clone - transcribe')
    task="transcribe"
    if voice is None:
        wprint("No audio file submitted! Please upload or record an audio file before submitting your request.")
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
    tprint(f'transcribe COMPLETE,{round(time2-time1,4)}s')
    tprint(f'  \nTranscribe result：\n 🔣Language：{language} \n 🔣Text：{text}' )
    return  text,language  

def clone_voice(user_voice,user_text,user_lang):
    if not duration(user_voice):
        return None
    if  user_text == '':
        wprint("Please enter text to generate/请输入生成文字")
        return None
    user_text=trim_text(user_text,user_lang)
    global gpt_path, sovits_path
    gpt_path = abs_path("pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
    #tprint(f'Model loaded:{gpt_path}')
    sovits_path = abs_path("pretrained_models/s2G488k.pth")
    #tprint(f'Model loaded:{sovits_path}')
    try:
        prompt_text, prompt_lang = transcribe(user_voice)
    except UnboundLocalError as e:
        wprint(f"The language in the audio cannot be recognized ：{str(e)}")
        return None
    tts_pipline.init_vits_weights(sovits_path)
    tts_pipline.init_t2s_weights(gpt_path)
    inputs={
        "text": user_text,
        "text_lang": dict_language[user_lang],
        "ref_audio_path": user_voice,
        "prompt_text": prompt_text,
        "prompt_lang": dict_language[prompt_lang],
        "top_k": 5,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut1",
        "batch_size":20,
        "speed_factor":1.0,
        "split_bucket":True,
        "volume":1.0,
        "return_fragment":False,
    }
  
    yield next(tts_pipline.run(inputs))

with open('dummy') as f:
    dummy_txt = f.read().strip().splitlines()

def dice():
    return random.choice(dummy_txt), '🎲'

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

with gr.Blocks(theme='Kasien/ali_theme_custom') as app:
    gr.HTML('''
  <h1 style="font-size: 25px;">TEXT TO SPEECH</h1>
  <h1 style="font-size: 20px;">Support English/Chinese/Japanese</h1>
  <p style="margin-bottom: 10px; font-size: 100%">
   If you like this space, please click the ❤️ at the top of the page..如喜欢，请点一下页面顶部的❤️<br>
  </p>''')

    gr.Markdown("""* This space is based on the text-to-speech generation solution [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) . 
    You can visit the repo's github homepage to learn training and inference.<br>
    本空间基于文字转语音生成方案 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). 你可以前往项目的github主页学习如何推理和训练。 
    * ⚠️Generating voice is very slow due to using HuggingFace's free CPU in this space. 
    For faster generation, click the Colab icon below to use this space in Colab,
    which will significantly improve the speed.<br>
    由于本空间使用huggingface的免费CPU进行推理，因此速度很慢，如想快速生成，请点击下方的Colab图标，
    前往Colab使用已获得更快的生成速度。
    <br>Colabの使用を強くお勧めします。より速い生成速度が得られます。 
    *  each model can speak three languages.<br>每个模型都能说三种语言<br>各モデルは3つの言語を話すことができます。""")   
    gr.HTML('''<a href="https://colab.research.google.com/drive/1fTuPZ4tZsAjS-TrhQWMCb7KRdnU8aF6j" target="_blank"><img src="https://camo.githubusercontent.com/dd83d4a334eab7ada034c13747d9e2237182826d32e3fda6629740b6e02f18d8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f6c61622d4639414230303f7374796c653d666f722d7468652d6261646765266c6f676f3d676f6f676c65636f6c616226636f6c6f723d353235323532" alt="colab"></a>
''')

    default_voice_wav, default_voice_wav_words, default_language, _, default_model_name, _, default_tone_sample_path = update_model("Trump")
    english_models = [name for name, _ in models_by_language["English"]]
    chinese_models = [name for name, _ in models_by_language["中文"]]
    japanese_models = [name for name, _ in models_by_language["日本語"]]
    with gr.Row():
        english_choice = gr.Radio(english_models, label="EN",value="Trump",scale=3)
        chinese_choice = gr.Radio(chinese_models, label="ZH",scale=2)
        japanese_choice = gr.Radio(japanese_models, label="JA",scale=4)

    plsh='Support【English/中文/日本語】，Input text you like / 輸入文字 /テキストを入力する'
    limit='Max 70 words. Excess will be ignored./单次最多处理120字左右，多余的会被忽略'

    gr.HTML('''
    <b>Input Text/输入文字</b>''')
    with gr.Row():
        with gr.Column(scale=2): 
            model_name = gr.Textbox(label="Seleted Model/已选模型", value=default_model_name, scale=1) 
            text_language = gr.Textbox(
            label="Language for input text/生成语言",
            info='Automatic detection of input language type.',scale=1,interactive=False
            ) 
        text = gr.Textbox(label="INPUT TEXT", lines=5,placeholder=plsh,info=limit,scale=10,min_width=0)
        ddice= gr.Button('🎲', variant='tool',min_width=0,scale=0)

        ddice.click(dice, outputs=[text, ddice])
        text.change( lang_detector, text, text_language)


    with gr.Row():
        with gr.Column(scale=2):    
            tone_select = gr.Radio(
            label="Select Tone/选择语气",
            choices=["tone1","tone2","tone3"],
            value="tone1",
            info='Tone influences the emotional expression ',scale=1)
        tone_sample=gr.Audio(label="🔊Preview tone/试听语气 ", scale=8)


    with gr.Accordion(label="prpt voice", open=False,visible=False):
        with gr.Row(visible=True):
            inp_ref = gr.Audio(label="Reference audio", type="filepath", value=default_voice_wav, scale=3)
            prompt_text = gr.Textbox(label="Reference text", value=default_voice_wav_words, scale=3)
            prompt_language = gr.Dropdown(label="Language of the reference audio", choices=["中文", "English", "日本語"], value=default_language, scale=1,interactive=False)
            dummy = gr.Radio(choices=["中文","English","日本語"],visible=False)
     
    
    with gr.Accordion(label="Additional generation options/附加生成选项", open=False):
        with gr.Row():
            how_to_cut = gr.Dropdown(
                label=("How to split input text?/如何对输入文字切片"),
                choices=[("Do not split/不切"), ("Split into groups of 4 sentences/四句一切"), ("Split every 50 characters/50字一切"), 
                         ("Split at CN/JP periods (。)/按中日文句号切"), ("Split at English periods (.)/按英文句号切"), ("Split at punctuation marks/按标点切"), ],
                value=("Split into groups of 4 sentences/四句一切"),
                interactive=True,
            info='A suitable splitting method can achieve better generation results/适合的切片方法会得到更好的效果'
            )
            split_bucket = gr.Checkbox(label="Split bucket/数据分桶", value=True, info='Speed up the inference process/提升推理速度')
        with gr.Row():
            volume = gr.Slider(minimum=0.5, maximum=5, value=1, step=0.1, label='Volume/音量',info='audio distortion due to excessive volume/大了要爆音')
            speed_factor = gr.Slider(minimum=0.25,maximum=4,step=0.05,label="Speed factor",value=1.0,info='Playback speed/播放速度')
            batch_size = gr.Slider(minimum=1,maximum=100,step=1,label="Batch size",value=20,info='The number of sentences for batch inference./并行推理的句子数量')
        with gr.Row():
            top_k = gr.Slider(minimum=1,maximum=100,step=1,label="top_k",value=5)
            top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label="top_p",value=1)
            temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label="temperature",value=1)
        ref_text_free = gr.Checkbox(label="REF_TEXT_FREE", value=False, visible=False)
        
        
    
    gr.HTML('''
    <b>Generate Voice/生成</b>''')
    with gr.Row():
        main_button = gr.Button("✨Generate Voice", variant="primary", scale=2)
        output = gr.Audio(label="💾Download it by clicking ⬇️", scale=6)
        #info = gr.Textbox(label="INFO", visible=True, readonly=True, scale=1)

    gr.HTML('''
    Generation is slower, please be patient and wait/合成比较慢，请耐心等待<br>
    If it generated silence, please try again./如果生成了空白声音，请重试
    <br><br><br><br>
    <h1 style="font-size: 25px;">Clone custom Voice/克隆自定义声音</h1>
    <p style="margin-bottom: 10px; font-size: 100%">Need 3~10s audio.This involves voice-to-text conversion followed by text-to-voice conversion, so it takes longer time<br>
    需要3~10秒语音，这个会涉及语音转文字，之后再转语音，所以耗时比较久
    </p>''')
    
    with gr.Row():
        user_voice = gr.Audio(type="filepath", label="（3~10s）Upload or Record audio/上传或录制声音",scale=3)
        with gr.Column(scale=7): 
            user_lang = gr.Textbox(label="Language/生成语言",info='Automatic detection of input language type.',interactive=False)
            with gr.Row():
                user_text= gr.Textbox(label="Text for generation/输入想要生成语音的文字", lines=5,placeholder=plsh,info=limit)
                dddice= gr.Button('🎲', variant='tool',min_width=0,scale=0)
       
        dddice.click(dice, outputs=[user_text, dddice])

    user_text.change( lang_detector, user_text, user_lang)

    user_button = gr.Button("✨Clone Voice", variant="primary")
    user_output = gr.Audio(label="💾Download it by clicking ⬇️")

    gr.HTML('''<div align=center><img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.laobi.icu/badge?page_id=Ailyth/DLMP9" /></div>''')
    
    english_choice.change(update_model, inputs=[english_choice], outputs=[inp_ref, prompt_text, prompt_language,dummy,model_name, tone_select, tone_sample])
    chinese_choice.change(update_model, inputs=[chinese_choice], outputs=[inp_ref, prompt_text, prompt_language, dummy,model_name, tone_select, tone_sample])
    japanese_choice.change(update_model, inputs=[japanese_choice], outputs=[inp_ref, prompt_text, prompt_language,dummy,model_name, tone_select, tone_sample])
    tone_select.change(update_tone, inputs=[model_name, tone_select], outputs=[inp_ref, prompt_text, tone_sample])
    
    main_button.click(
    inference,
    inputs=[text, 
              text_language,
              inp_ref, 
              prompt_text, 
              prompt_language,
              top_k, 
              top_p, 
              temperature, 
              how_to_cut, 
              batch_size, 
              speed_factor, 
              ref_text_free,
              split_bucket,
              volume],
    outputs=[output]
    )

    user_button.click(
    clone_voice,
    inputs=[user_voice,user_text,user_lang],
    outputs=[user_output])

app.launch(share=True, show_api=False).queue(api_open=False)