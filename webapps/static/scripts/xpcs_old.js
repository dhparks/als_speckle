
// declare all the variables
var gui = {};
gui.isLocked = false;

gui.sizes = {};
gui.sizes.x = 512;
gui.sizes.r = 30;
gui.sizes.s = 4;
gui.sizes.d = 5;

gui.intensity = {}
gui.intensity.images  = {};
gui.intensity.dataId  = null;
gui.intensity.regions = {};
gui.intensity.translateX = 0;
gui.intensity.translateY = 0;
gui.intensity.staticTranslateX = 0;
gui.intensity.staticTranslateY = 0;
gui.intensity.zoomFactor = 1.25;

gui.plots = {}
gui.plots.xScale   = null;
gui.plots.yScale   = null;
gui.plots.linefunc = null;
gui.plots.margins  = {top: 20, right: 20, bottom: 30, left: 50};
gui.plots.selectedPlot = null;
gui.plots.nframes = null;

var dragFunctions = {
    
    // methods for dragging the regions and background in the left svg
    regionBehavior: d3.behavior.drag()
            .origin(function() { var t  = d3.select(this); return {x: t.attr("x"),y: t.attr("y")};})
            .on("drag",   function()  { dragFunctions.regionDrag(this) })
            .on("dragend",function () { dragFunctions.regionDragEnd(this) }),

    regionDrag: function (x) {
	if (Object.keys(gui.backendTasks).length === 0) {
	    var t = d3.select(x);
            dragFunctions.updateBoxes(t.attr("regionId"),t.attr("location"))
	    }
         },
	 
    regionDragEnd: function (x) {
	
	if (Object.keys(gui.backendTasks).length === 0) {

	    // update the region information in the tracker
	    var id   = d3.select(x).attr("regionId");
	    var group = d3.select("#regionGroup"+id)
	    
	    r = gui.intensity.regions[id];
	    r.coords.rmin = parseInt(group.select('[location="mainRegion"]').attr("y"));
	    r.coords.cmin = parseInt(group.select('[location="mainRegion"]').attr("x"));
	    r.coords.rmax = parseInt(group.select('[location="lowerRight"]').attr("y"));
	    r.coords.cmax = parseInt(group.select('[location="lowerRight"]').attr("x"));   

	    // run the recalculation if that option is selected
	    if ($('#autocalc').is(":checked")) { userFunctions.recalculateG2() };
	}
    },
    
    // this is the drag behavior for the background image
    backgroundBehavior: d3.behavior.drag()
			.origin(function() { console.log('origin!');var t  = d3.select(this); return {x: t.attr("x"),y: t.attr("y")};})
			.on("drag", function() {
			    if (Object.keys(gui.backendTasks).length === 0 && !$('#lockBg').is(":checked")) {
				console.log("dragging 2")
				dragFunctions.updateBackground()}
			    })
			.on("dragend",function () {
			    // when dragging has finished, update the javascript information about the position
			    if (Object.keys(gui.backendTasks).length === 0 && !$('#lockBg').is(":checked")) {
				gui.intensity.staticTranslateX = gui.intensity.translateX;
				gui.intensity.staticTranslateY = gui.intensity.translateY;
				}
			    }),
    
    updateBackground: function () {

	// update the gui.translateX and gui.translateY parameters
	// with the information coming from the drag event; then
	// update the image transform properties
	
	var dx = d3.event.x, dy = d3.event.y;
	gui.intensity.translateX = gui.intensity.staticTranslateX+dx/gui.intensity.zoom;
	gui.intensity.translateY = gui.intensity.staticTranslateY+dy/gui.intensity.zoom;
	var str = "scale("+gui.intensity.zoom+") translate("+gui.intensity.translateX+","+gui.intensity.translateY+")"
	console.log(str)
	d3.select('#dataimage').attr("transform",str)
	
    },
    
    updateBoxes: function (regionId, what) {
    
        // this function updates the coordinates of the 6 svg elements which
        // show a region. most of the mathematical complexity arises from the
        // decision to enforce boundary conditions at the edge of the data, so
        // no aspect of the region becomes undraggable.

        var group = d3.select('#regionGroup'+regionId);
        
        var mr = group.select('[location="mainRegion"]');
        var ul = group.select('[location="upperLeft"]');
        var ur = group.select('[location="upperRight"]');
        var lr = group.select('[location="lowerRight"]');
        var ll = group.select('[location="lowerLeft"]');
        var cs = group.select('.selecter');
        var cw = parseInt(mr.attr("width"));
        var ch = parseInt(mr.attr("height"));
        
        // define shorthands so that lines don't get out of control
        var mx, my, ulx, uly, urx, ury, llx, lly, lrx, lry, csx, csy;
        var ds = gui.sizes.d, ss = gui.sizes.s, wh = gui.sizes.x, ww = gui.sizes.x;
        var dx = d3.event.x, dy = d3.event.y
	    
        // one behavior for dragging .main...
        if (what === 'mainRegion') {
    
            mrx = dx;      mry = dy;
            ulx = dx-ds;   uly = dy-ds;
            llx = dx-ds;   lly = dy+ch;
            urx = dx+cw;   ury = dy-ds;
            lrx = dx+cw;   lry = dy+ch;
            csx = dx+cw/2; csy = dy-ss;
    
            // now check their bounds
            if (ulx < 0) {mrx = ds; ulx = 0; llx = 0; urx = cw+ds; lrx = cw+ds;};
            if (uly < 0) {mry = ds; uly = 0; ury = 0; lly = ch+ds; lry = ch+ds;};
            if (urx > ww-ds) {mrx = ww-cw-ds; urx = ww-ds; lrx = ww-ds; ulx = ww-cw-2*ds; llx = ww-cw-2*ds;};
            if (lly > wh-ds) {mry = wh-ch-ds; uly = wh-ch-2*ds; ury = wh-ch-2*ds; lly = wh-ds; lry = wh-ds;};
            csx = mrx+cw/2; csy = mry-ss;
        }
        
        // ... another for dragging a corner
        else {
             // pull the old values
            x1 = parseInt(ul.attr("x"));
            x2 = parseInt(ur.attr("x"));
            y1 = parseInt(ur.attr("y"));
            y2 = parseInt(lr.attr("y"));
            
            // calculate bounding functions
            x1b = Math.min(x2-ds,Math.max(0,dx));
            x2b = Math.max(x1+ds,Math.min(ww-ds,dx));
            y1b = Math.min(y2-ds,Math.max(0,dy));
            y2b = Math.max(y1+ds,Math.min(wh-ds,dy));
            
            // calculate the new values
            if (what === 'upperLeft')  {new_x1 = x1b; new_x2 = x2; new_y1 = y1b; new_y2 = y2;}
            if (what === 'upperRight') {new_x1 = x1; new_x2 = x2b; new_y1 = y1b; new_y2 = y2;}
            if (what === 'lowerLeft')  {new_x1 = x1b; new_x2 = x2; new_y1 = y1; new_y2 = y2b;}
            if (what === 'lowerRight') {new_x1 = x1; new_x2 = x2b; new_y1 = y1; new_y2 = y2b;}
            var new_width  = new_x2-new_x1;
            var new_height = new_y2-new_y1;
            
            // assign the coordinates
            mrx = new_x1+ds; mry = new_y1+ds;
            ulx = new_x1;    uly = new_y1;
            urx = new_x2;    ury = new_y1;
            llx = new_x1;    lly = new_y2;
            lrx = new_x2;    lry = new_y2;
            csx = new_x1+(new_width+ds)/2; csy = new_y1
            
            mr.attr("width",new_width-ds).attr("height",new_height-ds)    
        }
        
        // update the positions
        mr.attr("x",mrx).attr("y",mry)
        ul.attr("x",ulx).attr("y",uly);
        ur.attr("x",urx).attr("y",ury);
        ll.attr("x",llx).attr("y",lly);
        lr.attr("x",lrx).attr("y",lry);
        cs.attr("cx",csx).attr("cy",csy);
    }
}
    
var userFunctions = {
    
    addRegion: function () {
	
	var createSVG = function (reg) {
        
	    // this method draws the region on the left SVG. only gets called once
	    // per region
	    
	    var tc = reg.coords;
	    
	    // make a group on the SVG for the 6 elements
	    var g = d3.select("#svgintensity").append("g").attr("id","regionGroup"+reg.regionId);
    
	    // define the relevant common attributes of the boxes
	    var rs = gui.sizes.r, ds = gui.sizes.d, ss = gui.sizes.s;
	    var allBoxes = [
		{h:rs, w: rs, x:tc.cmin,    y:tc.rmin,    c:"mainRegion", curs: "move"},
		{h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin+rs, c:"lowerRight", curs: "se-resize"},
		{h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin-ds, c:"upperRight", curs: "ne-resize"},
		{h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin+rs, c:"lowerLeft",  curs: "sw-resize"},
		{h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin-ds, c:"upperLeft",  curs: "nw-resize"}];
		
	    // make the rectangular elements using d3
	    for (var k=0;k<allBoxes.length;k++) {
		
		var thisBox = allBoxes[k];
		var newBox  = g.append("rect")
		
		newBox
		    .attr("x",thisBox.x)
		    .attr("y",thisBox.y)
		    .attr("height",thisBox.h)
		    .attr("width",thisBox.w)
		    .attr("regionId",reg.regionId)
		    .attr("location",thisBox.c)
		    .style("fill",reg.color)
		    .style("fill-opacity",0)
		    .style('cursor',thisBox.curs)
		    .call(dragFunctions.regionBehavior) // attaches the dragging behavior
		    ;
    
		if (thisBox.c==="mainRegion") {
		    newBox.style("stroke",reg.color)
		    .style("stroke-width",2)
		    .style("stroke-opacity",1);}
		    
		if (thisBox.c !="mainRegion") {
		    newBox.style("fill-opacity",1);}
	    }
		
	    // make the circular element
	    g.append("circle")
		.attr("cx",tc.cmin+rs/2)
		.attr("cy",tc.rmin-ss)
		.attr("r",ss)
		.style("fill",reg.color)
		.style("fill-opacity",0) // need a fill to toggle on clickSelect
		.style("stroke-width",2)
		.style("stroke",reg.color)
		.classed("selecter",true)
		.classed("region",true)
		.classed('interactive',true)
		.attr("regionId",reg.regionId)
		.on("click",function () {
		    var t = d3.select(this);
		    console.log("click")
		    console.log(t.attr("regionId"));
		    userFunctions.selectRegion(t.attr("regionId"));
		})
	};
	
	var newHue = function () {
        
	    var hues = [];
	    for (region in gui.intensity.regions) {hues.push(gui.intensity.regions[region].hue);}
	    hues.sort(function(a,b) {return a-b});
    
	    if (hues.length === 0) {return Math.random();}
	    if (hues.length === 1) {return (hues[0]+0.5)%1;}
	    if (hues.length === 2) {return hues[0]+Math.max((hues[0]-hues[1]+1)%1,(hues[1]-hues[0]+1)%1)/2;}
	    else {
		
		// find the biggest gap in the list of hues and put the new hue
		// in the middle of that gap. 
		
		var distances = [], gap, idx, hue;
		
		for (var n=0;n<hues.length-1;n++) {distances.push(hues[n+1]-hues[n])};
		distances.push(1+hues[0]-hues[hues.length-1]);
		
		gap = Math.max.apply(Math, distances)
		idx = distances.indexOf(gap);
		
		return (hues[idx]+gap/2+1)%1
	    }
	};

	var r = {};
	
	// initial coordinates
	r.coords = {
	    rmin:(gui.sizes.x-gui.sizes.r)/2,
	    rmax:(gui.sizes.x+gui.sizes.r)/2,
	    cmin:(gui.sizes.x-gui.sizes.r)/2,
	    cmax:(gui.sizes.x+gui.sizes.r)/2}
	
	// identifiers
	var x = new Date().getTime()
	x = x.toString()
	t = "r"+x.slice(x.length-7,x.length-2);
	console.log(t)
	console.log(typeof t)
	r.regionId = t;
	r.hue      = newHue()
	r.color    = d3.hsl(r.hue*360,1,0.5).toString();
	r.selected = false;
	
	// data and fit
	r.functional = null;
	r.g2Values   = [];
	r.fitValues  = [];
	r.fit        = {};
	
	// add to tracker
	gui.intensity.regions[t] = r;
	
	// draw the svg with drag actions attached
	createSVG(r);
	
	// send the information to the backend
	var backend = function (callback) {
	    
	    // send this array to the backend
	    var send = {uid:r.regionId,coords:[r.coords.rmin,r.coords.rmax,r.coords.cmin,r.coords.cmax]}
	    $.ajax({
		url: "xpcs/new",
		type: 'POST',
		data: JSON.stringify(send),
		contentType: 'application/json; charset=utf-8',
		dataType: 'json',
		async: true,
		success: function(data) {
		    console.log(data);
		    callback(null);
		}
	    });
	}

	var frontend = function (error) {
	    if (error != null) {console.log(error)}
            if ($('#autocalc').is(":checked")) { userFunctions.recalculateG2() };
	    };
	
	queue().defer(backend).await(frontend);
	
    },

    changeBackground: function () {
        // change the background to the permutation of selected colormap and selected scale
        var scale = $("input[name=scale]:checked").attr("id");
        var color = $("input[name=cm]:checked").attr("id");
	var path  = '/static/xpcs/images/data_session'+gui.sessionId+'_id'+gui.intensity.dataId+'_'+color+'_'+scale+'.jpg';
	var tfrm  = "scale("+gui.intensity.zoom+") translate("+gui.intensity.staticTranslateX+","+gui.intensity.staticTranslateY+")"
	d3.select('#dataimage').attr("xlink:href", path)
	d3.select("#dataimage").attr("transform",tfrm);
	console.log(tfrm);
    },
    
    convertCoord: function (val) {
	return {'x':(val-gui.intensity.staticTranslateX*gui.intensity.zoom)/gui.intensity.zoom,
		'y':(val-gui.intensity.staticTranslateY*gui.intensity.zoom)/gui.intensity.zoom}
    },
    
    deleteRegions: function () {
        
	// delete from the backend and the frontend all the selected regions.
	// if they don't exist on the backend, it doesnt matter
        var backend = function (callback) {
            
            var ts = new Date().getTime()
            gui.backendTasks[ts] = 'deleteRegions';
	    
	    $.ajax({
		url: "xpcs/remove",
		type: 'POST',
		data: JSON.stringify(selectedRegions),
		contentType: 'application/json; charset=utf-8',
		dataType: 'json',
		async: true,
		success: function(data) {
		    console.log(data);
		    delete gui.backendTasks[ts]
		    callback(null);
		    }
	    });
	};

        var frontend = function (error) {
            // tell the front end which regions to remove
            
            if (error != null) {
                console.log("error removing frontend");
                console.log(error);}

            // check if the group selection for the text display of the fit parameters
            // matches an element in selected. if so, delete the display
            if (selectedRegions.length > 0) {
                var fitGroup = d3.select("#fitParamsText").attr("selectedGroup");
                if (fitGroup != "none") {
                    var idx = selectedRegions.indexOf(fitGroup);
                    if (idx > -1) {userFunctions.textBoxTransition(0)}
                    gui.plots.selectedPlot = null;
                };
            };
                
	    for (var k = 0; k < selectedRegions.length; k++) {
		thisRegion = selectedRegions[k]
		if (gui.intensity.regions[thisRegion].selected) {
		    
		    // remove from dictionary
		    delete gui.intensity.regions[thisRegion]
		    
		    // remove group from intensity plot
		    d3.select("#regionGroup"+thisRegion).remove()
		    
		    // remove group from g2 plot
		    d3.select("#g2Group"+thisRegion).remove()
		}
	    }
        };
	
	// find all the selected regions
	var selectedRegions = [];
	for (region in gui.intensity.regions) {
	    if (gui.intensity.regions[region].selected) {
		selectedRegions.push(region)
	    }
	}
	
	// remove the selected regions from the backend, then from the frontend
        queue().defer(backend).await(frontend);

    },
    
    initGraph: function () {
        
        // define log scales for the new data
        gui.plots.xScale = d3.scale.log().range([0, gui.plots.width]).domain([1,gui.plots.nframes]);
        gui.plots.yScale = d3.scale.log().range([gui.plots.height, 0]).domain([1e-6,1]).clamp(true);
	
	// define the interpolation function for plotting
	gui.plots.linefunc = d3.svg.line()
                .interpolate("linear")
                .x(function(d) { return gui.plots.xScale(d.x); })
                .y(function(d) { return gui.plots.yScale(d.y); });
        
        var resetSVG = function () {
	    
            d3.select("plotGroup").remove();
            d3.select("#svggraphs")
                .append("g")
                .attr("transform", "translate(" + gui.plots.margins.left + "," + gui.plots.margins.top + ")")
                .attr("id","plotGroup");
        };
	
        var drawGrids = function () {
            //draw grid lines
            svgg.append("g").attr("id","verticalGrid")
            d3.select("#verticalGrid").selectAll(".gridlines")
                .data(gui.plots.xScale.ticks()).enter()
                .append("line")
                .attr("class","gridlines")
                .attr("x1",function (d) {return gui.plots.xScale(d)})
                .attr("x2",function (d) {return gui.plots.xScale(d)})
                .attr("y1",function ()  {return gui.plots.yScale(1e-6)})
                .attr("y2",function ()  {return gui.plots.yScale(1e0)})
    
            svgg.append("g").attr("id","horizontalGrid")
            d3.select("#horizontalGrid").selectAll(".gridlines")
                .data(gui.plots.yScale.ticks()).enter()
                .append("line")
                .attr("class","gridlines")
                .attr("x1",function ()  {return gui.plots.xScale(1)})
                .attr("x2",function ()  {return gui.plots.xScale(gui.plots.nframes)})
                .attr("y1",function (d) {return gui.plots.yScale(d)})
                .attr("y2",function (d) {return gui.plots.yScale(d)})
        };
                
        var drawAxes = function () {
            // draw axes
            
            // define xAxis and yAxis
            var nticks = Math.floor(Math.log(gui.plots.nframes)/Math.LN10);
            var xAxis  = d3.svg.axis().scale(gui.plots.xScale).orient("bottom").ticks(nticks);
            var yAxis  = d3.svg.axis().scale(gui.plots.yScale).orient("left");//.ticks(5);//.tickFormat(d3.format(".1f"));
            
            svgg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + gui.plots.height + ")")
                .call(xAxis);
          
            svgg.append("g")
                .attr("class", "y axis")
                .attr("transform","translate(0,0)")
                .call(yAxis)
                .append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", 6)
                .attr("dy", ".71em")
                .style("text-anchor", "end")
                .text("G2");
        };
        
        var drawReadout = function () {
            // set up the fit parameters readout
            d3.select("#plotGroup").append("g").attr("id","fitParamsGroup")
        
            d3.select("#fitParamsGroup")
                .append("rect")
                .attr("x",-5).attr("y",5)
                .attr("opacity",0)
                .attr("id","txtrect")
                .attr("fill","white")
                .style("stroke-width",1)
                .style("stroke","black")
                
            d3.select("#fitParamsGroup")
                .append("text")
                .attr("id","fitParamsText")
                .attr("selectedGroup","none");
            };
                   
        //
        resetSVG();
            
        var svgg = d3.select("#plotGroup")
        
        drawGrids()
        drawAxes()
        drawReadout()

        },
    
    recalculateG2: function () {

        var funcString, fileId

	// 1. send regions coordinates to the backend.
	// 2. backend calculates g2 in all regions which have changed.
	// 3. backend fits data in all regions which have changed
	// 4. backend returns as a json object the g2 and fit values
	// 5. g2 and fit get attached to gui.regions
	// 6. plots get redrawn
	
	var redraw = function (error) {
	    
	    if (error != null) { console.log(error) }
	    
	    // record which line is currently selected, if applicable
	    var oldSelection = gui.plots.selected;

	    // when redrawing, there is no such thing as a selected plot.
	    userFunctions.selectPlot(gui.plots.selected);
	    gui.plots.selected = null;

	    // the way the data is structured poses a problem for the d3 enter/exit
	    // methodology (ie, doesnt work!). instead it seems easier to simply
	    // remove all the children groups of #plotGroup and replot all data
	    svgg = d3.select('#plotGroup');
	
	    // clear all the old plots
	    svgg.selectAll(".dataSeries").remove()

	    // d3 doesn't play well with being passed an object instead of an
	    // array. therefore, recast gui.intensity.regions into an array
	    // of objects with only the necessary data
	    var plottables = []
	    for (var region in gui.intensity.regions) {
		var nr = {}, tr = gui.intensity.regions[region]
		nr.regionId  = tr.regionId;
		nr.color     = tr.color;
		nr.g2Values  = tr.g2Values;
		nr.fitValues = tr.fitValues;
		plottables.push(nr)
	    }
	    
	    // each plot group is structured as
	    // (parent) group
	    //    (child)  path g2 data
	    //    (childs) circles g2 data
	    //    (child)  path fit data

	    var newLines = svgg.selectAll(".dataSeries")
		.data(plottables)
		.enter()
		.append("g")
		.attr("class","dataSeries")
		.attr("id", function (d) {return "g2Group"+d.regionId;})
		.style("cursor","pointer")
		.on("click",function () {userFunctions.selectPlot(d3.select(this).attr("id"))});
	
	    newLines.append("path")
		.style("fill","none")
		.style("stroke-width",2)
		.style("stroke",function (d) {return d.color})
		.attr("class","dataPath")
		.attr("id",function(d) {return "g2Data"+d.regionId})
		.attr("d", function(d) {return gui.plots.linefunc(d.g2Values); });

	    // define a scale for the size of the circles so that they decrease in
	    // size as they approach 1
	    gui.plots.rScale = d3.scale.log().domain([1,gui.plots.nframes]).range([7,0]).clamp(false);
	
	    // add data circles.
	    for (var k=0;k<plottables.length;k++) {
		d3.select('#g2Group'+plottables[k].regionId).selectAll(".g2circle")
		    .data(plottables[k].g2Values)
		    .enter().append("circle")
		    .attr("class","g2circle")
		    .attr("cx",function (d) {return gui.plots.xScale(d.x)})
		    .attr("cy",function (d) {return gui.plots.yScale(d.y)})
		    .attr("r", function (d) {return gui.plots.rScale(d.x)})
		    .style("fill", plottables[k].color)
	    }

	    // add the fit lines. these are added last to ensure they are above the circles.
	    // in the future, it might be better to draw the fit lines after the click,
	    // so that they are on top of everything.
	    newLines.append("path")
		.style("fill","none")
		.style("stroke","black")
		.style("stroke-width",2)
		.style("opacity",0)
		.attr("class","fitPath")
		.attr("id",function(d) {return "g2Fit"+d.regionId})
		.attr("d", function(d) {return gui.plots.linefunc(d.fitValues); })
		;   
		
	    userFunctions.selectPlot(oldSelection);
	    
	    //move the parameters box to the top
	    var g = d3.select("#fitParamsGroup")[0][0]
	    g.parentNode.appendChild(g)
	    }
	
	var parse = function (data) {
	    
	    console.log(data)
	    
	    // take the data returned from the backend fitting and attach
	    // it to the correct locations in gui.intensity etc
	    var functional = data.fitting.functional;
	    var parameters = data.fitting.parameters;
	    
	    for (region in data.analysis) {
		
		thisData   = data.analysis[region];
		thisRegion = gui.intensity.regions[region];
		
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
	    }
	    
	}
	
	var backend = function (callback) {
	    
	    var _convertRegion = function (r) {
		out = {}
		out.rmin = userFunctions.convertCoord(r.rmin).y;
		out.rmax = userFunctions.convertCoord(r.rmax).y;
		out.cmin = userFunctions.convertCoord(r.cmin).x;
		out.cmax = userFunctions.convertCoord(r.cmax).x;
		return out
	    }
	    
	    // lock
	    var ts = new Date().getTime();
            gui.backendTasks[ts] = 'recalculatePlot'
	    
	    // change the cursor to "progress"
	    $("body").css("cursor","progress")

	    // loop over regions, building a coordinates array
	    data = {}
	    data.coords = {}
	    data.form   = $("input[name=fitform]:checked").val()
	    for (region in gui.intensity.regions) {
		data.coords[region] = _convertRegion(gui.intensity.regions[region].coords)}
	    
	    // send this array to the backend. when it comes back, parse it,
	    // then draw the plots
	    $.ajax({
		    url: "xpcs/calculate",
		    type: 'POST',
		    data: JSON.stringify(data),
		    contentType: 'application/json; charset=utf-8',
		    dataType: 'json',
		    async: true,
		    success: function(data) {
			parse(data);
			$("body").css("cursor","default")
			delete gui.backendTasks[ts]
			callback(null);
			}
		});
	}
	
	queue().defer(backend).await(redraw);
    },

    selectPlot:function (groupId) {
    
        //console.log("clicked on: "+groupId)
	//console.log("current:    "+gui.plots.selected)
    
        // selectors
        if (gui.plots.selected != null) {
	    var oldRegionGroup = "#regionGroup"+gui.plots.selected;
            var oldGroupId     = "#g2Group"+gui.plots.selected;
            var oldFitId       = "#g2Fit"+gui.plots.selected;}
	    
        if (groupId != null) {
            var regionId   = groupId.replace("g2Group","");
            var newGraphId = groupId;
            var newFitId   = "#g2Fit"+regionId;}
	    var newRegionGroup = '#regionGroup'+regionId
	    
        if (groupId === null) {regionId = null}

        // first, deselect the selected plot if the selected plot is not null
        if (gui.plots.selected != null ) {
            //console.log("turning off fit "+oldFitId)
            d3.select(oldFitId).transition().duration(150).style("opacity",0);
            d3.select(oldRegionGroup).select("[location=mainRegion]").transition().duration(150).style("fill-opacity",0);
            userFunctions.textBoxTransition(0,regionId);
            };
        
        // now select the desired plot, if: 1. the desired plot is different than
        // the currently selected plot and 2. the desired plot is not null.
        if (oldFitId != newFitId && regionId != null) {
            d3.select(newRegionGroup).select("[location=mainRegion]").transition().duration(150).style("fill-opacity",0.5)
            d3.select(newFitId).transition().duration(150).style("opacity",1);};
        
        var osp = gui.plots.selected;
        if (osp === regionId) {gui.plots.selected = null};
        if (osp  != regionId) {gui.plots.selected = regionId};
        
        // update the display of the fit parameters
        if (gui.plots.selected != null) {
            
	    var thisRegion = gui.intensity.regions[gui.plots.selected]
	    
	    // get the fit parameters
	    var fitmap = thisRegion.fitParamsMap;
	    var fitval = thisRegion.fitParamsVals;
	    var lines = ["id: "+thisRegion.regionId,thisRegion.functional]
	    for (var key in fitmap) {lines.push(fitmap[key]+": "+fitval[parseInt(key)].toPrecision(4))}
	    
            // remove the old box, then build the new box
            d3.selectAll(".fitText").remove();
            var txt = d3.select("#fitParamsText");
            txt.selectAll("tspan").data(lines).enter()
                .append("tspan")
                .text(function (d) {return d})
                .attr("class","fitText")
                .attr("x",0)
                .attr("dy","1.2em")
                .style("opacity",0);
            txt.attr("selectedGroup",regionId)
        
        //d3.select('#g2_'+regionId+"_fit").transition().duration(150).style("opacity",1);
        userFunctions.textBoxTransition(1,regionId);
        }

    },
    
    selectRegion: function (t) {
	
        // given the regionId "t", select the region in the tracker
	gui.intensity.regions[t].selected = !gui.intensity.regions[t].selected;
	
	// select the region in the svg; switch the opacity
	var c = d3.select("#regionGroup"+t).select("circle");
        var o = 1-c.style("fill-opacity");
        c.transition().duration(150).style("fill-opacity",o)
    },

    textBoxTransition: function (newOpacity,regionId) {
        if (regionId === null)
            {regionId = gui.plots.selected}
        if (newOpacity === 0) {
            d3.selectAll("#txtrect").transition().duration(150).style("opacity",0);
            d3.selectAll(".fitText").transition().duration(150).style("opacity",0).remove();
            d3.select("#fitParamsText").attr("selectedGroup","none");
            }
        if (newOpacity === 1) {
            var txt = d3.select("#fitParamsText");
            var bbox = txt.node().getBBox();
	    
	    var tbx = gui.plots.xScale(1)+10;
	    var tby = gui.plots.yScale(1e-6)-10-bbox.height;
	    
            d3.select("#fitParamsGroup").attr("transform","translate("+tbx+","+tby+")");
            d3.select("#txtrect").attr("width",bbox.width+10).attr("height",bbox.height);
            d3.selectAll(".fitText").transition().duration(150).style("opacity",1);
            d3.select("#txtrect").transition().duration(150).style("opacity",1);}
    },

    zoomIn: function () { if (!$('#lockBg').is(":checked")) {userFunctions.zoom(gui.intensity.zoomFactor)}},
    
    zoomOut: function () { if (!$('#lockBg').is(":checked")) {userFunctions.zoom(1./gui.intensity.zoomFactor)}},

    zoom: function(factor) {
	
	// the current center coordinates  given by:
	var cc = userFunctions.convertCoord(gui.sizes.x/2);

	// update the zoom factor
	gui.intensity.zoom *= factor;
	    
	// calculate the new translation values to keep the current center
	gui.intensity.staticTranslateX = gui.sizes.x/(2*gui.intensity.zoom)-cc.x;
	gui.intensity.staticTranslateY = gui.sizes.x/(2*gui.intensity.zoom)-cc.y;
	    
	// apply the updated transformation
	var str = "scale("+gui.intensity.zoom+") translate("+gui.intensity.staticTranslateX+","+gui.intensity.staticTranslateY+")"
	d3.select('#dataimage').attr("transform",str);

    }
    
    };

var start = function () {
    
    // this is code that should run when the script is done loading.

    // using jquery, attach actions to the interface buttons. some buttons can only
    // be pressed while the gui is unlocked, meaning there are no pending backend tasks
    // which still need to come in.
    $("#load_data").click(function() {if (!gui.isLocaked) {userFunctions.loadData()};});
    $("#new_region").click(function() {if (!gui.isLocked) {userFunctions.addRegion()}});
    $("#delete").click(function() {if (!gui.isLocked) {userFunctions.deleteRegions()};});
    $("#recalculate").click(function() {if (!gui.isLocked) {userFunctions.recalculateG2()};});
    $("#colormaps").children().click(function () {userFunctions.changeBackground();});
    $("#scales").children().click(function () {userFunctions.changeBackground();});
    $("#zoom_in").click(function () {userFunctions.zoomIn()});
    $("#zoom_out").click(function () {userFunctions.zoomOut()});
    $("#functionals").children().click(function () {if (!gui.isLocked) {userFunctions.recalculatePlot()};});

    // using d3, append the right things to the divs
    d3.select("#intensity")
        .append("svg")
        .attr("id","svgintensity")
        .attr("width",gui.sizes.x)
        .attr("height",gui.sizes.x);
        
    d3.select('#svgintensity')
        .append("image")
        .attr("id","dataimage")
        .call(dragFunctions.backgroundBehavior);
        
    // skeletonize the right svg plots
    // if a graph already exists, remove it
    d3.select('#svggraphs').remove()
    
    // margins for svggraphs
    width  = gui.sizes.x - gui.plots.margins.left - gui.plots.margins.right,
    height = gui.sizes.x - gui.plots.margins.top - gui.plots.margins.bottom;
         
    gui.plots.width  = width
    gui.plots.height = height
    
    // create the chart group.
    d3.select("#graphs").append("svg").attr("id","svggraphs")
        .attr("width",  width + gui.plots.margins.left + gui.plots.margins.right)
        .attr("height", height + gui.plots.margins.top + gui.plots.margins.bottom)
	
    // query the backend to get the dataid and the number of frames
    var backend = function (callback) {
    
	var gotBack = 0;
    
	$.getJSON(
            'xpcs/purge', {},
            function(returned) {gotBack += 1; if (gotBack === 2) {callback(null)};}
            );
    
	$.getJSON(
            'xpcs/query', {},
            function(returned) {
                gui.sessionId        = returned.sessionId;
		gui.intensity.dataId = returned.dataId;
		gui.intensity.size   = returned.size;
		gui.plots.nframes    = returned.nframes;
		gui.intensity.zoom   = gui.sizes.x/gui.intensity.size;
		gotBack += 1; if (gotBack === 2) { callback(null) };
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
		    img.src    = '/static/xpcs/images/data_session'+gui.sessionId+'_id'+gui.intensity.dataId+'_'+color+'_'+scale+'.jpg';
		}
	    }
	};

	var _setImage = function (error) {
	    d3.select("#dataimage").attr("width",gui.intensity.size).attr("height",gui.intensity.size)
	    userFunctions.changeBackground();
	    userFunctions.initGraph();
	}
	
	queue().defer(_downloadImages).await(_setImage)

    }
    
    queue().defer(backend).await(frontend);

};

start();