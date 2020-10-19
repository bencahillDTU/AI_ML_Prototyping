# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:12:23 2020

@author: benca
"""

import gtts
from playsound import playsound

tts = gtts.gTTS("Hello Skylab")
tts.save("hello.mp3")
playsound("hello.mp3")

