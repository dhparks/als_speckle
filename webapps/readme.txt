SPECKLE ANALYSIS SERVER

1. Purpose
The speckle analysis server is intended to fill a void in the BL12.0.2 user experience: users at the beamline are often unable to determine if the data they collect is any good until they return to their home institutions. As a result, extended periods of beamtime may be dedicated to the collection of garbage. The speckle analysis server addresses this shortcoming by providing a web-based graphical interface to a fast, gpu-equipped, remote server running an up-to-date installation of the speckle python library.

2. Components and additional installations

Speckle analysis server consists of four components:
    
    1. A set of HTML webpages and corresponding javascript applications which run the graphical interface
    2. A python webframework which routes user commands coming from the graphical interface to the analysis backends
    3. A set of analytical backends which manage and analyze data through calls to the speckle library
    4. The speckle library, which performs calculations
    
The HTML/javascript layer should run correctly in any modern browser; browsers tested are Chrome 30, Safari 5.1, and Firefox 23. No versions of IE have been tested for correct behavior. I haven't tested Safari 6 because my computer is too old and crappy for it.

The python webframework which functions as webserver and request router to the backends is contained in the file flask_server.py. Running this file requires the flask python library, which has a webpage at

http://flask.pocoo.org/

To install flask on a computer with pip (the python package manager), just run "sudo pip install Flask"

The analytical backends which manage the analysis are stored in speckle/interfaces and may be considered part of the speckle library. The speckle library is of course stored in speckle/ and has many modules.

3. Firewall configuration
When flask_server.py is run not in development/debugging mode, it responds to valid requests from all origins on port :5000. This is a security threat as data analysis requires data to be uploaded to the server. While safeguards exist within the server application to prevent the uploading of files which may contain executable code, the requirement to accept very large XPCS datasets makes the server susceptible to denial-of-service attacks via the upload mechanism. For this reason it is advisable to limit access to the server to the internal LBNL network. On magnon.lbl.gov, the development host of the speckle library, the built-in firewall accepts ssh and sftp requests from all sources, but allows access to port :5000, the default port for the analysis server, only from IP addresses within wired lbl.gov subnets. Port 5001 is the development server which is unstable and does not allow uploads. Here is a printout of the firewall status (sudo ufw status verbose) on Magnon

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
5001                       ALLOW IN    131.243.0.0/16
5001                       ALLOW IN    128.3.0.0/16
22                         ALLOW IN    Anywhere (v6)
115/tcp                    ALLOW IN    Anywhere (v6)

The range of lbl.gov IPs (v4) spans two subnets: 131.243.x.x and 128.3.x.x (cf https://commons.lbl.gov/display/itdivision/IP+Subnet+Addresses+at+LBNL). From other subnets, in particular LBNL wifi or networks outside LBNL, access to the server requires you to authenticate using VPN:

https://commons.lbl.gov/display/itdivision/VPN+-+Virtual+Private+Network

3. Accessing the Magnon server
Accessing the analysis development server is very simple. From a computer on the LBNL network or a computer on an external network with VPN access, direct a web browser to magnon.lbl.gov:5000 (or whatever computer you have the server running on). Keep in mind that this is a development server which may be turned off or modified at any time.

5. Running your own local server
A server for running your own analysis jobs or developing new analysis front ends for others to use is simple: just run "python flask_server.py" from a terminal window. Depending on your python installation you may need to change some path variables in the flask_server.py file so that it can find your installation of the speckle library. With an unmodified version of the code I have provided, your server will run in debug mode and accept only local connections at 127.0.0.1:5000 (or maybe localhost:5000). Running a server exposed to the broader network requires knowledge of how to secure your computer and for this reason I will not provide instructions, although they may be easily found in the Flask documentation (see link above).

6. Lifetime of sessions
Because the analytical server is intended to run continuously, allowing analysis sessions to persist indefinitely would ultimately consume infinite computer resources. For this reason, sessions time out after a specified life-time set by default as 8 hours. This lifetime is enforced in a lazy manner only when a new session is created, at which time the session manager examines current sessions for expiration and purges those which are too old; additionally, the server deletes corresponding intermediate results from the disk. Consequently, I strongly advise that users of the speckle server download their analysis once they are done. It seems unlikely that any use of the rapid-analysis tools will require more than 8 hours of work.

7. The demonstration server
A modification to flask_server.py is also provided to show the capabilities of the analysis without allowing users to upload data. Consequently, the server administrator (i.e., YOU!) must provide some data to play with. This modified server application is stored in demo_server.py and is run in exactly the same way as flask_server.py, with the exception that, by default, it accepts requests on port 5001. Data is served from the “demodata” directory. This functionality is provided to help new users get accustomed to the analysis functionality using standard datasets.

