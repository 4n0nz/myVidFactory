import whisper, json
model = whisper.load_model('medium')
result = model.transcribe('source_audio.wav', language='en', verbose=False)
segs = [{'start': round(s['start'],2), 'end': round(s['end'],2), 'text': s['text'].strip()} for s in result['segments']]
json.dump(segs, open('source_audio.json','w'), ensure_ascii=False, indent=2)
print(f'Done: {len(segs)} segments')
