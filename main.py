import os
import io
import re
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

# 1. Загружаем модель Silero при старте сервера
device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'v4_ru.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt', local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

class TTSRequest(BaseModel):
    text: str
    voice: str = "baya" # Шикарные женские голоса: baya, kseniya, xenia

# 2. Умная разбивка текста, чтобы не взорвать память Render'a
def split_text(text, max_len=600):
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    curr = ""
    for s in sentences:
        if len(curr) + len(s) < max_len:
            curr += s + " "
        else:
            if curr: chunks.append(curr.strip())
            curr = s + " "
    if curr: chunks.append(curr.strip())
    return chunks

@app.post("/tts")
def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    
    try:
        # Убираем спецсимволы
        clean_text = req.text.replace('*', '').replace('#', '').replace('~', '')
        chunks = split_text(clean_text)
        tensors = []
        
        # 3. Озвучиваем по одному предложению
        for chunk in chunks:
            if not chunk.strip(): continue
            # 24000 Гц для скорости, качество останется отличным
            audio_tensor = model.apply_tts(text=chunk, speaker=req.voice, sample_rate=24000)
            tensors.append(audio_tensor)
        
        if not tensors:
            raise ValueError("Нет аудио")
            
        # 4. Склеиваем всё в один большой аудиофайл
        final_tensor = torch.cat(tensors).unsqueeze(0)
        
        # 5. Пакуем в WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, final_tensor, 24000, format='wav')
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        print("Ошибка:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
