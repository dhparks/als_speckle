SPECKLE ANALYSIS SERVER

1. Purpose
The speckle analysis server is intended to fill a void in the BL12.0.2 user experience: users at the beamline are often unable to determine if the data they collect is any good until they return to their home institutions. As a result, extended periods of beamtime may be dedicated to the collection of garbage. The speckle analysis server addresses this shortcoming by providing a web-based graphical interface to a fast, gpu-equipped, remote server running an up-to-date installation of the speckle python library.

2. Components and additional installations

Speckle analysis server consists of four components:
    
    1. A set of HTML webpages and corresponding javascript applications which run the graphical interface
    2. A python webframework which routes user commands coming from the graphical interface to the analysis backend
    3. A set of analytical backends which manage and analyze data through calls to the speckle library
    4. The speckle library, which performs calculations
    
The HTML/javascript layer should run correctly in any modern browser; browsers tested are Chrome 30, Safari 5, and Firefox 23. No versions of IE have been tested for correct behavior. I haven't tested Safari 6 because my computer is too old and crappy for it.

The python webframework which functions as webserver and request router to the backends is contained in the file flask_server.py. Running this file requires the flask python library, which has a webpage at

http://flask.pocoo.org/

To install flask on a computer with pip, just run "sudo pip install Flask"

The analytical backends which manage the analysis are stored in speckle/interfaces and may be considered part of the speckle library. The speckle library is of course stored in speckle/ and has many modules.

3. Firewall configuration
When flask_server.py is run not in development/debugging mode, it responds to valid requests from all origins. This is a security threat as data analysis requires data to be uploaded to the server. While safeguards exist within the server application to prevent the uploading of files which may contain executable code, the requirement to accept very large XPCS datasets makes the server susceptible to denial-of-service attacks via the upload mechanism. For this reason it is advisable to limit access to the server to the internal LBNL network. On magnon.lbl.gov, the development host of the speckle library, the built-in firewall accepts ssh and sftp requests from all sources, but allows access to port :5000, the default port for the analysis server, only from IP addresses within wired lbl.gov subnets. Here is a printout of the firewall status (sudo ufw status verbose)

Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing)
New profiles: skip

To                         Action      From
--                         ------      ----
22                         ALLOW IN    Anywhere
115/tcp                    ALLOW IN    Anywhere
5000                       ALLOW IN    131.243.0.0/16
5000                       ALLOW IN    128.3.0.0/16
22                         ALLOW IN    Anywhere (v6)
115/tcp                    ALLOW IN    Anywhere (v6)

The range of lbl.gov IPs (v4) spans two subnets: 131.243.x.x and 128.3.x.x (cf https://commons.lbl.gov/display/itdivision/IP+Subnet+Addresses+at+LBNL). From other subnets, in particular LBNL wifi or networks outside LBNL, access to the server requires you to authenticate using VPN:

https://commons.lbl.gov/display/itdivision/VPN+-+Virtual+Private+Network

3. Accessing the Magnon server
Accessing the analysis development server is very simple. From a computer on the LBNL network or a computer on an external network with VPN access, direct a web browser to magnon.lbl.gov:5000 (or whatever computer you have the server running on). Keep in mind that this is a development server which may be turned off or modified at any time.

5. Running your own local server
A server for running your own analysis jobs or developing new analysis front ends for others to use is simple: just run "python flask_server.py" from a terminal window. Depending on your python installation you may need to change some path variables in the flask_server.py file so that it can find your installation of the speckle library. With an unmodified version of the code I have provided, your server will run in debug mode and accept only local connections at 127.0.0.1:5000 (or maybe localhost:5000). Running a server exposed to the broader network requires knowledge of how to secure your computer and for this reason I will not provide instructions, although they may be easily found in the Flask documentation (see link above).

6. Known issues
The outstanding known issue with the current server is its support of only a single connection. If multiple users attempt to access the analysis functions for the same project (for example, two experimenters try to do XPCS analysis at the same time) the data of both will become overwritten and the analysis will be nonsense. It is possible that I might fix this in the future by serving the application with a more robust server like gunicorn.
