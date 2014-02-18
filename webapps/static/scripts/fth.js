// front-end variables are stored in the "gui" object. each
// analysis "window" is stored in gui.components; in the
// fth analysis, there are 3 such windows
var gui = {};
gui.isLocked = false;
gui.components = {}
gui.components.hologram = {}
gui.components.acutance = {}
gui.components.propagation = {}

// configurable element sizes
gui.sizes = {};
gui.sizes.dragger = 5;
gui.sizes.region  = 30;
gui.sizes.window  = 300;
gui.sizes.selector = 4;
gui.sizes.scaler  = gui.sizes.window/300; // don't change this!!!!

// variables which describe the data. populated later.
gui.data = {};
gui.data.exists = false;

// locking functions
gui.lock = function ()   {$("body").css("cursor", "progress"); gui.isLocked = true;}
gui.unlock = function () {$("body").css("cursor", "default");  gui.isLocked = false;}

var guiFunctions = {
    
    actionDispatch: function (args) {
	// can eval handle this? if so, SHOULD it?
	if (args['action'] === 'zoomIn')    {gui.components[args['where']].background.zoomIn()}
	if (args['action'] === 'zoomOut')   {gui.components[args['where']].background.zoomOut()}
	if (args['action'] === 'lockBg')    {gui.components[args['where']].background.lock()}
	if (args['action'] === 'propagate') {guiFunctions.propagate(args)}
	if (args['action'] === 'scrub')     {gui.components.propagation.background.raster(args['n'])}
	if (args['action'] == 'showbp')     {guiFunctions.changeRasterBackground(args)};
	if (args['action'] == 'showgrad')   {guiFunctions.changeRasterBackground(args)};
    },
    
    validateForm: function (id) {
	var who  = document.getElementById(id);
	var what = who.value;
	if (isNaN(what))  {who.className="fieldr"};
	if (!isNaN(what)) {who.className="fieldg"};
	if (what==='')    {who.className="field0"};
    },
    
    changeRasterBackground: function (args) {
	var gcp = gui.components.propagation
	if (gcp.hasData) {
	    if (args.action !=  gcp.selected) {
		$("#"+args.where+'-'+args.action).css("fill","grey")
		try {$("#"+args.where+'-'+gcp.selected).css("fill","white")}
		catch (err) {}
		gcp.selected = args.action;}
	    var x = 'static/imaging/images/bp_session'+gui.data.sessionId+'_id'+gui.data.propagationId+'.jpg'
	    var imgPath = x.replace('bp',args.action.replace('show',''))
	    gcp.background.loadImage(imgPath)
	}
    },
    
    propagate: function (args) {
	// validate form. if its ok, send the parameters to python.
	// parse python's output and refresh the acutance graph.
	
	var _validateAndFormat = function() {
	    
	    // check value constraints
	    var e  = parseFloat($('#energy').val());
	    var p  = parseFloat($('#pitch').val());
	    var z1 = parseFloat($('#zmin').val());
	    var z2 = parseFloat($('#zmax').val());
	    var typesOK = !(isNaN(e) || isNaN(p) || isNaN(z1) || isNaN(z2))
	    
	    // reorder z values if necessary. this is for the gui only;
	    // will be used to set up the plot
	    var gcp = gui.components.propagation
	    var z3 = parseInt(z1);
	    var z4 = parseInt(z2);
	    if (z3 > z4) {gcp.zmin = z4; gcp.zmax = z3};
	    if (z4 > z3) {gcp.zmin = z3; gcp.zmax = z4};
	    if (z4 === z3) {typesOK = false};
	    
	    // format parameter dictionary
	    var ap = 0;
	    if ($('#apodize').is(":checked"))  {ap = 1;};
	    var gchr  = gui.components.hologram.regions
	    var info  = gchr[Object.keys(gchr)[0]].convertCoords();
	    var info2 = {'energy':e,'zmin':z1,'zmax':z2,'pitch':p,'apodize':ap};
	    for (var attrname in info2) {info[attrname] = info2[attrname]};
	    
	    info['check'] = (e != '' && p != '' && (z1 != '' || z1 === 0) && (z2 != '' || z2 === 0) && gui.data.exists && typesOK);
	    	
	    return info;
	}
	
	var _backend = function (callback) {
	    
	    var _success = function (json) {
		var gcp  = gui.components.propagation
		var gcpb = gui.components.propagation.background 
		gui.data.propagationId = json.propagationId;
		gcpb.frameSize         = json.frameSize;
		callback(null)
	    }
	    
	    // lock
	    gui.lock()
	    
	    // talk to python server; get propagationId. data is loaded in _frontend
	    $.getJSON("fth/propagate", info, _success);

	}
	
	var _frontend = function (error) {
	    
	    var _loadData   = function (callback) {
		
		var _parseData = function (error,data) {
		    
		    if (error != null) {console.log(error);}
		    
		    // parse the data and attach to acutance object
		    gca.data = data.map(function (d) {return {x:parseFloat(d.z),y:parseFloat(d.acutance)}})
		    gcp.hasData = true;
		    
		    // define new scales for the plots
		    gca.domainMin = gcp.zmin;
		    gca.domainMax = gcp.zmax;
		    gca.rangeMin  = 0;
		    gca.rangeMax  = 1;
		    gca.xScale    = d3.scale.linear().range([0, gca.width]).domain([gcp.zmin,gcp.zmax]);
		    gca.yScale    = d3.scale.linear().range([gca.height,0]).domain([gca.rangeMin,gca.rangeMax]);
		    gca.lineFunc  = d3.svg.line().interpolate("linear").x(function(d) { return gca.xScale(d.x); }).y(function(d) { return gca.yScale(d.y); });
		    callback(null);
		}

		// get the csv off the server, then parse it
		queue().defer(d3.csv, csvPath).await(_parseData);
	    }
	    
	    var _redraw = function (error) {
		if (error != null) { console.log(error) }
		gca.graph.draw({'yText':'Acutance','xText':'Distance'}); gca.graph.plot()

		// load a new image into the rasterBackground
		if (gcp.selected === null) {var action = 'showbp'}
		else {var action = gcp.selected}
		guiFunctions.actionDispatch({'where':'propagation','action':action})
	    }
	    
	    // load the data and the raster image, then redraw the
	    // acutance graph
	    var gca     = gui.components.acutance
	    var gcp     = gui.components.propagation
	    var csvPath = '/static/imaging/csv/acutance_session'+gui.data.sessionId+'_id'+gui.data.propagationId+'.csv'
	    var imgPath = 'static/imaging/images/bp_session'+gui.data.sessionId+'_id'+gui.data.propagationId+'.jpg'
	    queue().defer(_loadData).awaitAll(_redraw)
	    gui.unlock()
	}

	var info = _validateAndFormat();
	if (info.check) {queue().defer(_backend).await(_frontend)}
    },
}

var start = function () {

    var startHologram = function () {

	var h    = "hologram"
	
	// instantiate the draggable background
	var path = 'static/imaging/images/ifth_session'+gui.data.sessionId+'_id'+gui.data.dataId+'_'+'0.8_logd.jpg'
	var dbg  = new draggableBackground(h,gui.sizes.window,gui.sizes.window);
	dbg.draw()
	dbg.loadImage(path)
	gui.components[h].background = dbg;
	
	// create the clickable svg buttons. arguments: action, coords {x, y}
	// these do not need to be stored in the gui object
	var b = []
	b.push(new actionButton(h, 'zoomIn',     {x:5, y:5},  'plus'  ))
	b.push(new actionButton(h, 'zoomOut',    {x:30, y:5}, 'minus' ))
	b.push(new actionButton(h, 'lockBg',     {x:55, y:5}, 'lock'  ))
	b.push(new actionButton(h, 'propagate',  {x:275,y:5},'rArrow'))
	for (var k = 0; k < b.length; k++) {b[k].draw()} 
	
	// draw the draggable region
	gui.components[h].regions = {}
	reg = new draggableRegionWhite(h,gui.sizes,false)
	gui.components[h].regions[reg.regionId] = reg;
	reg.draw()
    }
	
    var startPropagation = function () {
	
	var gcp = gui.components.propagation
	gcp.background = new rasterBackground('propagation',gui.sizes.window,gui.sizes.window);
	gcp.background.draw()
	
	var b = [];
	b.push(new actionButton('propagation', 'showbp', {x:5, y:5},  'tophat'))
	b.push(new actionButton('propagation', 'showgrad', {x:30, y:5}, 'derivative'))
	for (var k = 0; k < b.length; k++) {b[k].draw()}
	
	gcp.selected = null;
	gcp.hasData  = false;
	gcp.rasterValue = 0;
	
    }

    var startAcutance = function () {

	// this stuff is populated here instead of in guiObjects because its size
	// is interface specific. guiObjects will look for this when it draws
	// the objects
	x         = {}
	x.margins = {top: 20, right: 30, bottom: 30, left: 50};
	x.width   = 2*gui.sizes.window+4-x.margins.left-x.margins.right;
	x.height  = 260-x.margins.bottom-x.margins.top;
	
	gui.components.acutance       = x;
	gui.components.acutance.graph = new sliderGraph("acutance",gui.sizes.window,gui.sizes.window)
    };
    
    var backend = function (callback) {
	// query the backend and get the dataid, sessionid, etc
	$.getJSON("fth/query", {},
	    function (returned) {
		// copy the data from returned into gui.data
		Object.keys(returned).forEach(function(key) {gui.data[key] = returned[key]});
		gui.data.exists = true;
		callback(null);
	    }
	)};
	
    var frontend = function (error) {
	startHologram();
	startPropagation();
	startAcutance();
    };
    
    queue().defer(backend).await(frontend);
};

start()