SPECKLE ANALYSIS SERVER

1. Purpose
The speckle analysis server is intended to fill a void in the BL12.0.2 user experience: users at the beamline are often unable to determine if the data they collect is any good until they return to their home institutions. As a result, extended periods of beamtime may be dedicated to the collection of garbage. The speckle analysis server addresses this shortcoming by providing a web-based graphical interface to a fast, gpu-equipped, remote server running an up-to-date installation of the speckle python library.

2. Components and additional installations

Speckle analysis server consists of four components:
    
    1. A set of HTML webpages and corresponding javascript applications which run the graphical interface
    2. A python webframework which routes user commands coming from the graphical interface to the analysis backend
    3. A set of analytical backends which manage and analyze data through calls to the speckle library
    4. The speckle library, which performs calculations
    
The HTML/javascript layer should run correctly in any modern browser. Firefox and Webkit-based browswers are preferred, as no versions of IE have been tested for correct behavior.

The python webframework which functions as webserver and request router is contained in the file flask_server.py. Running this file requires the flask python library, which has a webpage at

http://flask.pocoo.org/

To install flask on a computer with pip, just run "sudo pip install Flask"

The analytical backends which manage the analysis are stored in speckle/interfaces and may be considered part of the speckle library. The speckle library is of course stored in speckle/ and has many modules.

3. Firewall configuration
When flask_server.py is run not in development/debugging mode, it responds to valid requests from all origins. This is a security threat as data analysis requires data to be uploaded to the server. While safeguards exist within the server application to prevent the uploading of files which may contain executable code, the requirement to accept very large XPCS datasets makes the server susceptible to denial-of-service attacks via the upload mechanism. For this reason it is advisable to limit access to the server to the internal LBNL network. On magnon.lbl.gov, the development host of the speckle library, the built-in firewall accepts ssh and sftp requests from all sources, but allows access to port :5000, the default port for the analysis server,  only from IP addresses within the lbl.gov subnet. Here is a printout of the firewall status (sudo ufw status verbose)

Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing)
New profiles: skip

To                         Action      From
--                         ------      ----
22                         ALLOW IN    Anywhere
115/tcp                    ALLOW IN    Anywhere
5000                       ALLOW IN    131.243.0.0/16
22                         ALLOW IN    Anywhere (v6)
115/tcp                    ALLOW IN    Anywhere (v6)

The range of lbl.gov IPs (v4) is 131.243.0.0 through 131.243.255.255. Presumably, it is still possible to access the speckle analysis server from outside the lbl.gov subnet through the use of a VPN.

3. Running the server
As far as I know, running a persistent server is either: 1. hard or 2: annoying. Currently, the method of running flask_server on magnon is the following:

    1. Log in to magnon
    2. Switch to the screen multiplexer through the command "screen"
    3. Start the server through python flask_server.py
    4. Exit the screen multiplexer through (CTRL-A)+D
    5. Log out of magnon
    
This keeps the server running in a user account. When the server inevitably crashes due to some bug, this procedure must be repeated to restart it. Another possibility is running the server as an automatically (re)starting daemon/background process, but this is currently beyond my abilities.

4. Accessing the server
Accessing the analysis server is very simple. From a computer on the LBNL network or a computer on an external network with VPN access, direct a web browser to magnon.lbl.gov:5000 (or whatever computer you have the server running on)


    
    


