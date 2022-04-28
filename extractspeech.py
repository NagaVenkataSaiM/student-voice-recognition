import torch
import sys
 
print("Code by Nvsai") 
# loading vad model and tools to work with audio
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)



(get_speech_ts_adaptive, save_audio, read_audio,utils_vad,collect_chunks) = utils
 
audio = read_audio('voice_originals/mono/'+sys.argv[1])
 
 
# get time chunks with voice
speech_timestamps = get_speech_ts_adaptive(audio, model)
 
 
# gather the chunks and save them to a file
save_audio('voice_files/'+sys.argv[2],
         collect_chunks(speech_timestamps, audio)) 
