
config = {
    'verbose': False,
    'audio_file': 'goforward.raw',
    'hmm': get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\Models_Sphinx\cmusphinx-en-us-ptm-5.2'),
    'lm': get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\Models_Sphinx\en-70k-0.1.lm'),
    'dict': get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\Models_Sphinx\cmudict.hub4.06d.dict')
}

audio, sr = librosa.load(r'C:\Users\DELL\PycharmProjects\FYP_proj\03-01-05-02-01-01-22.wav', sr=16000)


decoder = pocketsphinx.Decoder(config)

decoder.start_utt()

for i in range(0, len(audio), 2048):
    decoder.process_raw(audio[i:i+2048], False, False)

decoder.end_utt()

hypothesis = decoder.hyp()
phonemes = [seg.word for seg in decoder.seg()]

print(phonemes)



# Create a Audio File object using the acoustic and language models
audio = AudioFile(
    verbose=False,
    buffer_size=2048,
    audio_file=r'C:\Users\DELL\PycharmProjects\FYP_proj\03-01-05-02-01-01-22.wav',
    no_search=False,
    full_utt=False,
    hmm=get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\venv\Lib\site-packages\pocketsphinx\model\en-us\en-us'),
    lm=get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\venv\Lib\site-packages\pocketsphinx\model\en-us\en-us-phone.lm.bin'),
    dict=get_model_path(r'C:\Users\DELL\PycharmProjects\FYP_proj\venv\Lib\site-packages\pocketsphinx\model\en-us\cmudict-en-us.dict')
)