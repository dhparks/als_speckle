from flask import Flask, jsonify, request, redirect, send_from_directory, url_for, json
from werkzeug import secure_filename
import os
app = Flask(__name__)
import time

import sys
sys.path.insert(0,'../pycommon')
import speckle, numpy

# if everything is present for a gpu, turn it on
try:
    import string
    import pyopencl
    import pyopencl.array as cla
    import pyfft
    use_gpu = True
except ImportError:
    use_gpu = False
    
if use_gpu: gpu_info = speckle.gpu.init()
else: gpu_info = None

# load the backends
from speckle.interfaces import xpcs_backend
from speckle.interfaces import imaging_backend
backendx = xpcs_backend.backend(gpu_info=gpu_info)
backendi = imaging_backend.backend(gpu_info=gpu_info)

# functions to handle file uploading, mostly just taken from flask online documents
def allowed_file(name):
    return '.' in name and name.rsplit('.', 1)[1] in allowed_exts

@app.route('/upload',methods=['GET','POST'])
def upload_file():

    if request.method == 'POST':
        
        project = request.files.keys()[0]
        
        print "loading"
        file = request.files[project]
        print "done"
        
        success = False
        
        if file and allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], '%s_data.fits'%project))
            
            if project in ('cdi','fth'):
                backend = backendi
            if project in ('xpcs',):
                backend = backendx
                backend.regions = {}
            backend.load_data("data/%s_data.fits"%project,project)
            
            success = True

        if success:
            return redirect('/'+project)

# the rest of the decorators are switchboard functions which take a request
# and send it to the correct backend
@app.route('/')
def serve_landing():
    return send_from_directory(".","landing.html")

@app.route('/xpcs')
def serve_xpcs():
    # serve the static page for xpcs
    backendx.regions = {}
    return send_from_directory('.', 'xpcs.html')

@app.route('/fth')
def serve_fth():
    # serve the static page for fth
    return send_from_directory('.','fth.html')

@app.route('/cdi')
def serve_cdi():
    # serve the static page for fth
    return send_from_directory('.','cdi.html')

@app.route('/xpcs/<cmd>',methods=['GET','POST'])
def xpcs_cmd(cmd):

    if cmd == 'remove':
        
        # json is just a list of numbers
        uids = request.json
        print uids
        
        print "regions before"
        print backendx.regions.keys()
        
        # remove them one at a time
        for uid in uids:
            try: del backendx.regions[int(uid)]
            except KeyError: pass
            
        print "regions after"
        print backendx.regions.keys()
            
        # return a json response
        return jsonify(result="removed")
    
    if cmd == 'query':
        return jsonify(dataId=backendx.data_id,nframes=backendx.frames)
    
    if cmd == 'new':
        
        r = request.json
        backendx.update_region(r['uid'],r['coords'])
        return jsonify(result='added region with uid %s'%backendx.newest)
    
    if cmd == 'purge':
        # reset the list of regions
        backendx.regions = {}
        return jsonify(result="regions purged")

    if cmd == 'calculate':
        
        # change the update the coordinates of all the regions. if they have
        # changed, this is noticed in the backend.
        
        # update the fit form
        try:
            form = request.json['form']
            if form != backendx.form and form in ('decayexp','decayexpbeta'):
                backendx.form    = form
                backendx.refitg2 = True 
        except KeyError: pass
        
        # update the coordinates
        ckeys = ('rmin','rmax','cmin','cmax')
        coords = request.json['coords']
        for uid in coords.keys():
            tc = coords[uid]
            backendx.update_region(int(uid),[int(tc[ckey]) for ckey in ckeys])
            
        print "regions:"
        print backendx.regions.keys()

        # calculate g2 and fit to form
        backendx.calculate()
        backendx.refitg2 = False
        
        # dump a result file
        backendx.csv_output()
        
        # build and return a json response. the response is a dictionary of dictionaries
        # structured as follows:
        # {fitting: {functional:functional, parameters:params_map}
        # analysis: {
        # uid1: {g2: g2_values, fit:fit_values, params:fit_parameters},
        # uid2: {g2: g2_values, fit:fit_values, params:fit_parameters}
        # etc}}
        # so the two top level keys are "functional" and "analysis"
        
        response = {}
        response['fitting']  = {'functional':backendx.functional,'parameters':backendx.fit_keys}
        response['analysis'] = {}
        
        for region in backendx.regions.keys():
            tmp = {}
            tmp['g2']     = backendx.regions[region].g2.tolist()
            tmp['fit']    = backendx.regions[region].fit_vals.tolist()
            tmp['params'] = backendx.regions[region].fit_params.tolist()
            
            response['analysis'][region] = tmp
        
        return json.dumps(response)

    if cmd == 'recalculate':
        # recalculate g2 for all specified regions; update the plot
        print "recalculating"
        
        # see if the functional form has changed
        form = request.args.get('form',0,type=str)
        print form
        if form in ('decayexp','decayexpbeta') and form != backendx.form:
            backendx.form = form
            backendx.refitg2 = True
            
        
            
        # recalculate g2 and fit; save output
        backendx.calculate()
        backendx.csv_output()
        
        # reset the refit property
        backendx.refitg2 = False
        
        # return a json response
        return jsonify(result='dumped files with timestamp %s'%backendx.file_id,fileId=backendx.file_id,functionalString=backendx.functional)

@app.route('/fth/<cmd>',methods=['GET','POST'])
def fth_cmd(cmd):
    print cmd
    
    if cmd == 'query':
        # return the information the frontend needs to pull images etc
        return jsonify(dataId=backendi.data_id,zooms=backendi.zooms,hasgpu=use_gpu)

    if cmd == 'propagate':

        # get the coordinates
        int_keys = ('zoom','apodize')
        flt_keys = ('rmin','rmax','cmin','cmax','zmin','zmax','energy','pitch')
        
        params = {}
        for key in int_keys: params[key] = request.args.get(key,0,type=int)
        for key in flt_keys: params[key] = request.args.get(key,0,type=float)

        # run the propagation
        backendi.propagate(params,'fth')
        return jsonify(result="propagation finished",propagationId=backendi.bp_id)

@app.route('/cdi/<cmd>',methods=['GET','POST'])
def cdi_cmd(cmd):
    
    if cmd == 'query':
        return jsonify(dataId=backendi.data_id,zooms=backendi.zooms,hasgpu=use_gpu)

    if cmd == 'load':
        print request.args.get('file',0,type=str)
        name = request.args.get('file',0,type=str).replace('C:\\fakepath\\','')
        size = request.args.get('resize',0,type=int)
        blocker = request.args.get('blocker',0,type=float)
        if name != 0: backendi.load_data('./data/%s'%name,'cdi',size,blocker)
        if name == 0: backendi.load_data('./data/test_mag.fits')
        return jsonify(result="data loaded",dataId=backendi.data_id,zooms=backendi.zooms)
    
    if cmd == 'makesupport':
        backendi.make_support(request.json)
        print "making support"
        return jsonify(result=str(numpy.sum(backendi.support)))
    
    if cmd == 'reconstruct':
        
        print request.args
        
        # passed params: iterations, numtrials, ismodulus, sigma, threshold
        int_keys = ('iterations','numtrials','ismodulus')
        flt_keys = ('sw_cutoff','sw_sigma')
        params = {}
        for key in int_keys: params[key] = request.args.get(key,0,type=int)
        for key in flt_keys: params[key] = request.args.get(key,0,type=float)
        
        backendi.reconstruct(params)
        return jsonify(rId=backendi.r_id,rftf=backendi.rftfq[::4].tolist())
    
    if cmd == 'propagate':

        # get the coordinates
        int_keys = ('apodize',)
        str_keys = ('round',)
        flt_keys = ('rmin','rmax','cmin','cmax','zmin','zmax','energy','pitch')
        
        params = {}
        for key in int_keys: params[key] = request.args.get(key,0,type=int)
        for key in flt_keys: params[key] = request.args.get(key,0,type=float)
        for key in str_keys: params[key] = request.args.get(key,0,type=str)
        
        print "prop params fs: "
        print params

        # run the propagation
        backendi.propagate(params,'cdi')
        return jsonify(result="propagation finished",propagationId=backendi.bp_id)
        
upload_folder = './data'
allowed_exts  = set(['fits',])
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 1024**3

if __name__ == '__main__':
    app.run(host="0.0.0.0")
    