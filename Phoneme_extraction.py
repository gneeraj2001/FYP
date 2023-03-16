import pocketsphinx
import librosa
import numpy as np
from pocketsphinx import AudioFile, get_model_path, Decoder
from pydub import AudioSegment

audio_file_path = r'C:\Users\DELL\PycharmProjects\FYP_proj\03-01-05-02-01-01-22.wav'
sound = AudioSegment.from_wav(audio_file_path)  # can do same for mp3 and other formats

raw = sound._data  # returns byte string
raw = bytes(raw)
#print(raw)


audio_file_path = r'C:\Users\DELL\PycharmProjects\FYP_proj\03-01-05-02-01-01-22.wav'
config = {
    'verbose': False,
    'hmm': get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\venv\Lib\site-packages\pocketsphinx\model\en-us\en-us'),
    'lm': get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\venv\Lib\site-packages\pocketsphinx\model\en-us\en-us-phone.lm.bin'),
    'dict': get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\venv\Lib\site-packages\pocketsphinx\model\en-us\cmudict-en-us.dict')

}

decoder = Decoder(config)

audio, sr = librosa.load(r'C:\Users\DELL\PycharmProjects\FYP_proj\03-01-05-02-01-01-22.wav', duration=2, offset=0.5 ,sr=16000)
audio_raw = bytes(audio)
decoder.start_utt()
decoder.process_raw(audio_raw, False, False)
decoder.end_utt()

hypothesis = decoder.hyp()
print(hypothesis.hypstr)
'''
# Open the WAV file
with open(audio_file_path, 'rb') as f:
    # Decode the audio file
    decoder.start_utt()
    while True:
        buf = f.read(1024)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            break
    decoder.end_utt()
'''
#decoder.start_utt()
#decoder.process_raw(audio, False, False)
#decoder.end_utt()

hypothesis = decoder.hyp()
print(hypothesis.hypstr)

'''
decoder.start_utt()
for phrase in raw:
    decoder.process_raw(phrase, False, False)
decoder.end_utt()
'''
#hypothesis = decoder.hyp()
#phonemes = [seg[0] for seg in hypothesis.best().iter_phones()]

print('done')


