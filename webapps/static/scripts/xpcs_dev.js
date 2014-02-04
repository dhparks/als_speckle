
// declare all the variables
var gui = {};
gui.isLocked = false;


gui.data = {}
gui.data.exists = false;

// configurable element sizes
gui.sizes = {};
gui.sizes.dragger = 5;
gui.sizes.region  = 30;
gui.sizes.window  = 512;
gui.sizes.selector = 4;
gui.sizes.scaler  = 1//gui.sizes.window/300; // don't change this!!!!

gui.components = {}
gui.components.intensity = {}
gui.components.intensity.images  = {};
gui.components.intensity.regions = {};

gui.components.plots = {}
gui.components.plots.selectedPlot = null;

var guiFunctions = {
    
    actionDispatch: function (args) {
	if (args['action'] == 'regionDragEnd') {
	    if ($('#autocalc').is(":checked")) {
		guiFunctions.recalculateG2(args)
		};
	    };
	if (args['action'] == 'zoomIn')       {gui.components[args['where']].background.zoomIn()};
	if (args['action'] == 'zoomOut')      {gui.components[args['where']].background.zoomOut()};
	if (args['action'] == 'lockBg')       {gui.components[args['where']].background.lock()};
	if (args['action'] == 'addRegion')    {guiFunctions.addRegion()};
	if (args['action'] == 'delRegions')   {guiFunctions.deleteRegions();};
	if (args['action'] == 'recalculate')  {guiFunctions.recalculateG2();};
	if (args['action'] == 'togglePlot')   {guiFunctions.toggleSelectedPlot(args)}
    },
    
    addRegion: function () {
	// draw a new draggable region on the intensity dbg
	reg = new draggableRegionColor("intensity",gui.sizes,true)
	gui.components.intensity.regions[reg.regionId] = reg;
	reg.draw()
    },
    
    changeBackground: function () {
	// change the background to the permutation of selected colormap and selected scale
        var scale = $("input[name=scale]:checked").attr("id");
        var color = $("input[name=cm]:checked").attr("id");
	var path  = '/static/xpcs/images/data_session'+gui.data.sessionId+'_id'+gui.data.dataId+'_'+color+'_'+scale+'.jpg';
	gui.components.intensity.background.loadImage(path)
    },
    
    deleteRegions: function () {
        
	// delete from the backend and the frontend all the selected regions.
	// if they don't exist on the backend, it doesnt matter
        var backend = function (callback) {
            //gui.lock()
	    $.ajax({
		url: "xpcs/remove",
		type: 'POST',
		data: JSON.stringify(selectedRegions),
		contentType: 'application/json; charset=utf-8',
		dataType: 'json',
		async: true,
		success: function(data) {callback(null);}
	    });
	};

        var frontend = function (error) {
            // tell the front end which regions to remove
            
            if (error != null) { console.log("error removing frontend"); console.log(error);}

	    var gcp = gui.components.plots
	    var gci = gui.components.intensity
	    
            // check if the group selection for the text display of the fit parameters
            // matches an element in selected. if so, delete the display
            if (selectedRegions.length > 0) {
                var fitGroup = gcp.selectedPlot;
                if (fitGroup != null) {
                    var idx = selectedRegions.indexOf(fitGroup);
                    if (idx > -1) {gcp.textBox.hide()}
                    gcp.selectedPlot = null;
                };
            };
            
	    // delete in multiple places
	    for (var k = 0; k < selectedRegions.length; k++) {
		thisRegion = selectedRegions[k]
		if (gci.regions[thisRegion].selected) {
		    delete gci.regions[thisRegion]
		    delete gcp.regions[thisRegion]
		    d3.select("#intensity-region-"+thisRegion).remove()
		    d3.select("#plots-g2group-"+thisRegion).remove()
		}
	    }
        };
	
	// find all the selected regions
	var selectedRegions = [];
	var gcir = gui.components.intensity.regions
	for (region in gcir) {
	    if (gcir[region].selected) {
		selectedRegions.push(region)
	    }
	}

	// remove the selected regions from the backend, then from the frontend
        queue().defer(backend).await(frontend);
    },
    
    recalculateG2: function () {

	var _getInfo = function () {
	    var data    = {};
	    data.form   = $("input[name=fitform]:checked").val()
	    
	    var gcir = gui.components.intensity.regions;
	    var k    = Object.keys(gcir);
	    data.coords = {};
	    for (j in k) {data.coords[k[j]] = gcir[k[j]].convertCoords()};
	    return data;
	};
	
	var _parse = function (data) {
	    
	    // take the data returned from the backend fitting and attach
	    // it to the correct locations in gui.components.plots etc
	    // data is attached to plots, NOT intensity! intensity has
	    // no use for huge strings of numbers!
	    var functional = data.fitting.functional;
	    var parameters = data.fitting.parameters;
	    
	    for (region in data.analysis) {

		// assign new region
		thisData   = data.analysis[region];
		gui.components.plots.regions[region] = {}
		thisRegion = gui.components.plots.regions[region];
		
		// copy g2 data and fit data. for plotting, g2 and fit
		// must be a list of objects {x:datax, y:datay}
		var g2s = []; fit = [];
		for (var k=0;k<thisData.g2.length;k++) {
		    g2s.push({y:thisData.g2[k], x:k+1})
		    fit.push({y:thisData.fit[k],x:k+1}) 
		}
		thisRegion.g2Values  = g2s;
		thisRegion.fitValues = fit;
		
		// copy functional and parameters
		thisRegion.functional    = functional;
		thisRegion.fitParamsMap  = parameters;
		thisRegion.fitParamsVals = thisData.params;
		
		// copy identifiers
		thisRegion.regionId = region;
		thisRegion.color    = gui.components.intensity.regions[region].color;

	    }
	    
	}
	
	var _backend = function (callback) {
	    // lock the gui
	    //gui.lock()
    
	    // get the region coord information
	    var info = _getInfo()
    
	    // send info the backend; parse and replot
	    $.ajax({
		    url: "xpcs/calculate",
		    type: 'POST',
		    data: JSON.stringify(info),
		    contentType: 'application/json; charset=utf-8',
		    dataType: 'json',
		    async: true,
		    success: function(data) {
			_parse(data);
			callback(null);
			}
		});
	};
	
	var _frontend = function(error) {

	    // if there is a plot selected, deselect it (but keep track for later)
	    var gcp = gui.components.plots;
	    var oldSelection = gui.components.plots.selectedPlot;
	    guiFunctions.toggleSelectedPlot({'id':oldSelection});

	    // replot the data
	    gcp.graph.replot()
	    
	    // reselect the previously-selected data
	    guiFunctions.toggleSelectedPlot({'id':oldSelection});
	    
	    //gui.unlock()
	}
	
	queue().defer(_backend).await(_frontend)

    },

    toggleSelectedPlot:function (args) {

	var gci  = gui.components.intensity
	var gcp  = gui.components.plots

	if (args['id'] === null) {regionId = null}
	else {regionId = args['id'].replace('plots-g2group-','')}

	if (gcp.selectedPlot != null) {
	    var oldRegionId = '#intensity-region-'+gcp.selectedPlot;
	    var oldGroupId  = '#plots-g2group-'+gcp.selectedPlot;}
	    
	if (regionId != null) {
	    var newRegionId = "#intensity-region-"+regionId;
	    var newGroupId  = "#plots-g2group-"+regionId;}
	
	var _deselectOld = function () {
	    if (gcp.selectedPlot != null ) {
		d3.select(oldGroupId+" .fitPath").style("opacity",0);
		gci.regions[gcp.selectedPlot].toggleFill();
		};
	    if (gcp.selectedPlot === regionId) {gcp.selectedPlot = null;}
	};
	
	var _selectNew = function () {
	    if (oldGroupId != newGroupId) {
		gci.regions[regionId].toggleFill()
		d3.select(newGroupId+" .fitPath").style("opacity",1);
		gcp.selectedPlot = regionId;
	    };
	};
	
	var _updateReadout = function () {
	     // update the display of the fit parameters
	    if (gcp.selectedPlot != null) {
		// make the lines for the update
		var thisRegion = gcp.regions[gcp.selectedPlot]
		var fitmap = thisRegion.fitParamsMap;
		var fitval = thisRegion.fitParamsVals;
		var lines  = ["id: "+thisRegion.regionId,thisRegion.functional+"  "]
		for (var key in fitmap) {lines.push(fitmap[key]+": "+fitval[parseInt(key)].toPrecision(4)+"  ")}
		gcp.textBox.update(lines)
	    }
	    
	    if (gcp.selectedPlot == null) {
		gcp.textBox.hide();}
	}
	    
	_deselectOld();
	_selectNew();
	_updateReadout();
    }
};

var start = function () {

    var attachActions = function () {
	// using jquery, attach actions to the interface buttons. some buttons can only
	// be pressed while the gui is unlocked, meaning there are no pending backend tasks
	// which still need to come in.
	$("#colormaps").children().click(function () {guiFunctions.changeBackground();});
	$("#scales").children().click(function () {guiFunctions.changeBackground();});
	$("#functionals").children().click(function () {if (!gui.isLocked) {userFunctions.recalculatePlot()};});
    }
    
    var startIntensity = function () {
	
	var i = "intensity"
	
	// instantiate the draggable background
	var dbg  = new draggableBackground(i,gui.sizes.window,gui.sizes.window);
	gui.components[i].background = dbg;
	gui.components[i].isLocked = false;
	dbg.draw()
	
	// create the clickable svg buttons. arguments: action, coords {x, y}
	// these do not need to be stored in the gui object
	var b1 = new actionButton(i, 'zoomIn',     {x:10, y:10},                  'plus')
	var b2 = new actionButton(i, 'zoomOut',    {x:35, y:10},                  'minus')
	var b3 = new actionButton(i, 'lockBg',     {x:60, y:10},                  'lock')
	var b4 = new actionButton(i, 'addRegion',  {x:10, y:gui.sizes.window-25}, 'square')
	var b5 = new actionButton(i, 'delRegions', {x:35, y:gui.sizes.window-25}, 'x')
	var b6 = new actionButton(i, 'recalculate',{x:gui.sizes.window-25, y:10}, 'rArrow')
	b1.draw(); b2.draw(); b3.draw(); b4.draw(), b5.draw(), b6.draw()

    }
    
    var startPlots = function () {
	// this stuff is populated here instead of in guiObjects because its size
	// is interface specific. guiObjects will look for this when it draws
	// the objects
	
	var gcp = gui.components.plots
	
	gcp.margins = {top: 20, right: 20, bottom: 30, left: 50};
	gcp.width   = gui.sizes.window-gcp.margins.left-gcp.margins.right;
	gcp.height  = gui.sizes.window-gcp.margins.top-gcp.margins.bottom;

	// define new scales for the plots
	gcp.domainMin = 1;
	gcp.domainMax = gui.data.nframes;
	gcp.rangeMin  = 1e-6;
	gcp.rangeMax  = 1;
	gcp.xScale    = d3.scale.log().range([0, gcp.width]).domain([gcp.domainMin,gcp.domainMax]);
	gcp.yScale    = d3.scale.log().range([gcp.height,0]).domain([gcp.rangeMin,gcp.rangeMax]).clamp(true);
	gcp.lineFunc  = d3.svg.line().interpolate("linear").x(function(d) { return gcp.xScale(d.x); }).y(function(d) { return gcp.yScale(d.y); });
	gcp.rScale    = d3.scale.log().domain([1,gui.data.nframes]).range([7,0]).clamp(false);
	gcp.regions   = {};
	
	// draw the plot
	gcp.graph = new clickerGraph("plots",gui.sizes.window,gui.sizes.window)
	gcp.graph.draw()
	
	// draw the readout
	gcp.textBox = new clickerReadout("plots")
    }

    // query the backend to get the dataid and the number of frames
    var backend = function (callback) {
    
	var gotBack = 0;
	var _check = function () {gotBack += 1; if (gotBack === 2) {callback(null)}}
    
	$.getJSON(
            'xpcs/purge', {},
            function(returned) {_check()}
            );
    
	$.getJSON(
            'xpcs/query', {},
            function(returned) {
		// copy the data from returned into gui.data
		Object.keys(returned).forEach(function(key) {gui.data[key] = returned[key]});
		gui.data.exists = true;
		_check();
                }
            );
        };
	
    // download the data jpgs
    var frontend = function (error) {
        
	var _downloadImages = function (callback) {
	
	    // calculate the number of color map permutations.
	    var colors   = $("[name='cm']").map(function () {return this.id}).get();
	    var scales   = $("[name='scale']").map(function () {return this.id}).get();
	    var permutes = colors.length*scales.length;
	    
	    // for each permutation, issue a GET to the server. the browswer will cache
	    // the image. when the image is downloaded, increment the downloaded counter
	    var downloaded = 0;
	    for (var cm = 0; cm < colors.length; cm++) {
		for (var s = 0; s < scales.length; s++) {
		    color      = colors[cm];
		    scale      = scales[s]
		    var img    = new Image();
		    img.onload = function () {downloaded += 1; if (downloaded === permutes) { callback(null); }}
		    img.src    = '/static/xpcs/images/data_session'+gui.data.sessionId+'_id'+gui.data.dataId+'_'+color+'_'+scale+'.jpg';
		}
	    }
	};

	var _setImage = function (error) {
	    guiFunctions.changeBackground();
	}
	
	attachActions();
	startIntensity()
	startPlots()
	queue().defer(_downloadImages).await(_setImage)
    }
    queue().defer(backend).await(frontend);
};

start();