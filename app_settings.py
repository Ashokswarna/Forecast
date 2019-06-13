# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:40:33 2019

@author: ashok.swarna
"""
import os
os.getcwd()
from configparser import ConfigParser


def read_config(filename='config.ini', section='settings'):
    os.chdir('C:/Users/ashok.swarna/Downloads/config')
    parser = ConfigParser()
    parser.read(filename)
    configurations = {}

    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            configurations[item[0]] = item[1]
    else:
        raise Exception('{0} not found in {1} file'.format(section, filename))
    return configurations

project_home = u'C:\\Users\\ashok.swarna\\Downloads'