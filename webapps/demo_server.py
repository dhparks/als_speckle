from flask import Flask, jsonify, request, redirect, send_from_directory, json, session, escape
from werkzeug import secure_filename
import os
app = Flask(__name__)

import time
from datetime import timedelta
import sys
sys.path.insert(0,'../pycommon')
import speckle, numpy
import shutil

# if everything is present for a gpu, turn it on
try:
    import string
    import pyopencl
    import pyopencl.array as cla
    import pyfft
    use_gpu = True
except ImportError:
    use_gpu = False
    
# code for backends; gets re-instantiated for each session
from speckle.interfaces import xpcs_backend_dev
from speckle.interfaces import imaging_backend

# concurrent user sessions are managed through a dictionary which holds
# the backends and the time last seen
sessions = {}
def manage_session():
    # see if there is currently a session attached to the incoming request.
    # if not, assign one. the way to check for a session is to try to get a
    # key; an error indicates no session
    
    def _delete_old_files(ct):
        
        # delete old files (currently they live for 8 hours)
        import glob
        life_hours = 8
        files  = glob.glob('static/*/images/*session*.*')
        files += glob.glob('static/*/csv/*session*.*')
        files += glob.glob('data/*session*.*')
        kept, deleted = 0, 0
        kept_sessions = []
        del_sessions  = []
        for f in files:
            
            try: session_id = int(f.split('_')[1].split('session')[1])
            except ValueError: session_id = int(f.split('_')[1].split('session')[1].split('.')[0])
            if ct-session_id > 10*60*60*life_hours:
                os.remove(f)
                deleted += 1
                del_sessions.append(session_id)
            else: 
                kept += 1
                kept_sessions.append(session_id)
                
        del_sessions  = set(del_sessions)
        kept_sessions = set(kept_sessions)
        print "kept %s files from %s sessions"%(kept,len(kept_sessions))
        print "deleted %s files from %s sessions"%(deleted,len(del_sessions))
        
    def _delete_old_sessions():
        # delete old sessions from the sessions dictionary
        tx = time.time()
        session_life_hours = 8
        for sk in sessions.keys():
            if tx-sessions[sk]['last'] > 60*60*session_life_hours:
                del sessions[sk]
                
    def _make_new_session():
        
        # make a new session
        t    = int(time.time()*10)
        s_id = str(t)[-8:]
        t2   = int(s_id)
        
        # spin up a new gpu context and new analysis backends
        if use_gpu: gpu_info = speckle.gpu.init()
        else: gpu_info = None
        
        backendx = xpcs_backend_dev.backend(session_id=s_id,gpu_info=gpu_info)
        backendi = imaging_backend.backend(session_id=s_id,gpu_info=gpu_info)
    
        # store these here in python; can't be serialized into the cookie!
        sessions[s_id]             = {}
        sessions[s_id]['backendx'] = backendx
        sessions[s_id]['backendi'] = backendi
        sessions[s_id]['last']     = time.time()
    
        # store these in the cookie?
        session.permanant = True
        session['s_id']   = s_id
        print "session %s"%s_id
        
        return t2
        
    try:
        s_id = session['s_id']
    except KeyError:
        ct = _make_new_session()
        _delete_old_files(ct)
        _delete_old_sessions()

# functions to handle file uploading, mostly just taken from flask online documents
def allowed_file(name):
    return '.' in name and name.rsplit('.', 1)[1] in allowed_exts

@app.route('/demo',methods=['GET',])
def upload_file():

    # get (or make) the session id
    manage_session()
    s_id = session['s_id']

    # make an id for the data
    t    = int(time.time()*10)
    d_id = str(t)[-8:]
    
    # get the project id from the request
    project = request.args['project']
    
    # move the right file from the demo folder to the upload folder
    # then run the backends in the same way as flask_server
    old_name = 'demodata/%s.fits'%project
    new_name = '%s/%sdata_session%s.fits'%(app.config['UPLOAD_FOLDER'],project,s_id)
    shutil.copy(old_name,new_name)
    
    print "data copied"
    
    # load the data into the correct backend
    if project in ('cdi','fth'):
        backend = sessions[s_id]['backendi']
    if project in ('xpcs',):
        backend = sessions[s_id]['backendx']
        backend.regions = {}
                
    backend.load_data(project,app.config['UPLOAD_FOLDER'])
    
    print "about to redirect to %s"%project
    
    return redirect('/'+project)

# the rest of the decorators are switchboard functions which take a request
# and send it to the correct backend
@app.route('/')
def serve_landing():
    # now send the landing page
    return send_from_directory(".","static/html/demo.html")

@app.route('/xpcs')
def serve_xpcs():
    print "in /xpcs"
    # serve the static page for xpcs
    s_id = session['s_id']
    sessions[s_id]['backendx'].regions = {}
    return send_from_directory('.', 'static/html/xpcs_dev.html')

@app.route('/fth')
def serve_fth():
    # serve the static page for fth
    return send_from_directory('.','static/html/fth.html')

@app.route('/cdi')
def serve_cdi():
    # serve the static page for fth
    return send_from_directory('.','static/html/cdi.html')

@app.route('/expired')
def serve_expired():
    return send_from_directory('.','static/html/expired.html')

@app.route('/xpcs/<cmd>',methods=['GET','POST'])
def xpcs_cmd(cmd):
    
    # update the session with the current time
    try:
        s_id = session['s_id']
        backend = sessions[s_id]['backendx']
        sessions[s_id]['last'] = time.time()
    except KeyError:
        return redirect('/expired')

    if cmd == 'remove':
        
        # json is just a list of numbers
        uids = request.json
        print uids
        
        # remove them one at a time
        for uid in uids:
            try: del backend.regions[str(uid)]
            except KeyError: pass
            
        # return a json response
        return jsonify(result="removed")
    
    if cmd == 'query':
        return jsonify(sessionId=backend.session_id,dataId=backend.data_id,nframes=backend.frames,size=backend.data_size)
    
    if cmd == 'new':
        r = request.json
        backend.update_region(r['uid'],r['coords'])
        return jsonify(result='added region with uid %s'%backend.newest)
    
    if cmd == 'purge':
        # reset the list of regions
        backend.regions = {}
        return jsonify(result="regions purged")

    if cmd == 'calculate':
        
        # change the update the coordinates of all the regions. if they have
        # changed, this is noticed in the backend.
        
        # update the fit form
        try:
            form = request.json['form']
            if form != backend.form and form in ('decayexp','decayexpbeta'):
                backend.form    = form
                backend.refitg2 = True 
        except KeyError: pass
        
        # update the coordinates
        ckeys = ('rmin','rmax','cmin','cmax')
        coords = request.json['coords']
        for uid in coords.keys():
            tc = coords[uid]
            backend.update_region(str(uid),[int(tc[ckey]) for ckey in ckeys])

        # calculate g2 and fit to form
        backend.calculate()
        backend.refitg2 = False
        
        # dump a result file
        backend.csv_output()
        
        # build and return a json response. the response is a dictionary of dictionaries
        # structured as follows:
        # {fitting: {functional:functional, parameters:params_map}
        # analysis: {
        # uid1: {g2: g2_values, fit:fit_values, params:fit_parameters},
        # uid2: {g2: g2_values, fit:fit_values, params:fit_parameters}
        # etc}}
        # so the two top level keys are "functional" and "analysis"
        
        response = {}
        response['fitting']  = {'functional':backend.functional,'parameters':backend.fit_keys}
        response['analysis'] = {}
        
        for region in backend.regions.keys():
            tmp = {}
            tmp['g2']     = backend.regions[region].g2.tolist()
            tmp['fit']    = backend.regions[region].fit_vals.tolist()
            tmp['params'] = backend.regions[region].fit_params.tolist()
            
            response['analysis'][region] = tmp
        
        return json.dumps(response)

@app.route('/fth/<cmd>',methods=['GET','POST'])
def fth_cmd(cmd):
    
    try:
        s_id = session['s_id']
        backend = sessions[s_id]['backendi']
        sessions[s_id]['last'] = time.time()
    except KeyError:
        return redirect('/expired')
    
    if cmd == 'query':
        
        # front end wants:
        # 1. dataId
        # 2. sessionId
        # 3. hasGPU

        # return the information the frontend needs to pull images etc
        return jsonify(sessionId=backend.session_id,dataId=backend.data_id,hasGPU=use_gpu)

    if cmd == 'propagate':

        # get the coordinates
        int_keys = ('zoom','apodize','window')
        flt_keys = ('rmin','rmax','cmin','cmax','zmin','zmax','energy','pitch')
        
        params = {}
        for key in int_keys: params[key] = request.args.get(key,0,type=int)
        for key in flt_keys: params[key] = request.args.get(key,0,type=float)

        # run the propagation
        backend.propagate(params,'fth')
        return jsonify(result="propagation finished",propagationId=backend.bp_id,frameSize=backend.imgx)

@app.route('/cdi/<cmd>',methods=['GET','POST'])
def cdi_cmd(cmd):
    
    try:
        s_id = session['s_id']
        backend = sessions[s_id]['backendi']
        sessions[s_id]['last'] = time.time()
    except KeyError:
        return redirect('/expired')
    
    if cmd == 'download':
        r_id = request.args.get('reconstructionId',0,type=str)
        backend.save_reconstruction(r_id)
        return jsonify(result="saved")

    if cmd == 'query':
        print "querying"
        print "session %s"%backend.session_id
        print "data    %s"%backend.data_id
        return jsonify(sessionId=backend.session_id,dataId=backend.data_id,zooms=backend.zooms,hasgpu=use_gpu)
    
    if cmd == 'makesupport':
        backend.make_support(request.json)
        return jsonify(result=str(numpy.sum(backend.support)))
    
    if cmd == 'reconstruct':
        
        # passed params: iterations, numtrials, ismodulus, sigma, threshold
        int_keys = ('iterations','numtrials','ismodulus')
        flt_keys = ('sw_cutoff','sw_sigma')
        params = {}
        for key in int_keys: params[key] = request.args.get(key,0,type=int)
        for key in flt_keys: params[key] = request.args.get(key,0,type=float)
        
        backend.reconstruct(params)
        return jsonify(rId=backend.r_id,rftf=backend.rftfq[::4].tolist())
    
    if cmd == 'propagate':

        # get the coordinates
        int_keys = ('apodize',)
        str_keys = ('round',)
        flt_keys = ('rmin','rmax','cmin','cmax','zmin','zmax','energy','pitch')
        
        params = {}
        for key in int_keys: params[key] = request.args.get(key,0,type=int)
        for key in flt_keys: params[key] = request.args.get(key,0,type=float)
        for key in str_keys: params[key] = request.args.get(key,0,type=str)

        # run the propagation
        backend.propagate(params,'cdi')
        return jsonify(result="propagation finished",propagationId=backend.bp_id)
        
upload_folder = './data'
allowed_exts  = set(['fits',])
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 1024**3 # maximum file size in bytes

# for session management
import os
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=60*8)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5001,debug=True)
    
