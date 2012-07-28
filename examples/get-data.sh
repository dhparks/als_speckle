#!/bin/sh
# Enthought distributes a curl in /Libraray/Frameworks/EPD, but it does not
# appear to work.  Default to using system cURL.
/usr/bin/curl -O http://dl.dropbox.com/u/27355570/data.zip
unzip data.zip -d data
