#!/usr/bin/env python

import os
import webbrowser


url = "file://%s/docs/build/html/index.html"
url = url % (os.path.dirname(os.path.abspath(__file__)))
webbrowser.open(url, new=0, autoraise=True)
