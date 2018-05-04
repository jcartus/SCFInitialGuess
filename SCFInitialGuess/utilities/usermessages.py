"""This module contains functions to log information and print 
user messages

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import warnings
from datetime import datetime

try:
    input = raw_input
except:
    pass

class MarkerTheme(object):
    """Parent class for marker themes, Abstract class!"""

    _error = ""
    _warning = ""
    _info_low = ""
    _info_medium = ""
    _info_high = ""
    _input = ""
    _line_continuation = ""



    def __init__(self):
        raise NotImplementedError("MarkerTheme is an abstract class!")

    @classmethod
    def error(cls):
        return cls._error + " "
    
    @classmethod
    def warning(cls):
        return cls._warning + " "

    @classmethod
    def info(cls, level):
        """Info marker for given level: 0 lowest, 2 highest"""
        return [
            cls._info_low + " ", 
            cls._info_medium + " ", 
            cls._info_high + " "
        ][level]

    @classmethod
    def input(cls):
        return cls._input + " "

    @classmethod
    def line_continuation(cls):
        return cls._line_continuation + " "

class BracketMarkers(MarkerTheme):
    """A marker theme for logging that uses brackes"""

    _error = "[X]"
    _warning = "[w]"

    _info_low = "[ ]"
    _info_medium = "[-]"
    _info_high = "[+]"
    
    _input = "[?]"

    _line_continuation = "..."


    def __init__(self):
        raise NotImplementedError("BracketMarkers is a static class!")


class Messenger(object):
    """
    Messenger (static class). Prints information to user prompt.

    Attributes:
        marker_theme: defines the theming of the markers.
        print_level: defines how much should be printed (0: nothing, 1: only 
        errors, and high importance info, 2: like 1 but also medium importance 
        info and warnings, 3: all)
    """

    marker_theme = BracketMarkers
    print_level = 3
    indentation_marker = " " * 21 

    @classmethod
    def parse(cls, message):
        """Make sure line breaks etc. have proper indentation"""
        
        return message.replace(
            "\n", 
            "\n" + cls.indentation_marker + cls.marker_theme.line_continuation()
        )

    @classmethod
    def time(cls):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "

    @classmethod
    def info(cls, message, level=0):
        """Write a user info to prompt, if it is important enough (specifed by 
        print_level)"""

        if cls.print_level == 3 or (cls.print_level == 2 and level >= 1) or \
            (cls.print_level == 1 and level >= 2):
            print(cls.marker_theme.info(level) + cls.time() + cls.parse(message))

    @classmethod
    def warn(cls, message, *args, **kwargs):
        """Raise a warning if printlevel is 2 or higher"""

        if cls.print_level > 1:
            print(cls.marker_theme.warning() + cls.time() + cls.parse(message))

    @classmethod
    def error(cls, message, *args, **kwargs):
        """Write /raise? an error"""
        warnings.warn(
            cls.marker_theme.error() + cls.time() + cls.parse(message), *args, **kwargs
        )

    @classmethod
    def input(cls, message):
        """Prompt user input """
        return input(cls.marker_theme.input() + cls.time() + cls.parse(message) + ": ")


