# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:53:50 2019

@author: ashok.swarna
"""

def segment_string(s):
    segment_len = 2
    return [s[i:i+segment_len] for i in range(len(s) - (segment_len - 1))]



a = segment_string('abcdef')