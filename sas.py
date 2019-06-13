# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:16:52 2019

@author: ashok.swarna
"""

def segment_string(s):
    segment_len = 2
    A = [s[i:i+segment_len] for i in range(len(s) - (segment_len - 1))]
    del A[1]
    del A[3]
    segment_len = 3
    B = [s[i:i+segment_len] for i in range(len(s) - (segment_len - 1))]
    del B[1]
    del B[2]
    A.append(B)
    return A



a = segment_string("abcdef")