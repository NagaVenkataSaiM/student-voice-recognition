import sys
from pydub import AudioSegment
mysound=AudioSegment.from_wav("voice_originals/"+sys.argv[1])
mysound.export("voice_originals/mono/"+sys.argv[2],format="wav")
print("Code by Nvsai M")
