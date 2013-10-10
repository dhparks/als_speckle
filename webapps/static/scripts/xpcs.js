
// declare all the variables
var front = {};
front.backendTasks = {};

front.sizes = {};
front.sizes.x = 512;
front.sizes.r = 30;
front.sizes.s = 4;
front.sizes.d = 5;

front.intensity = {}
front.intensity.image   = null;
front.intensity.dataId  = null;
front.intensity.regions = {};

front.plots = {}
front.plots.xScale   = null;
front.plots.yScale   = null;
front.plots.linefunc = null;
front.plots.margins  = {top: 20, right: 20, bottom: 30, left: 50};
front.plots.selectedPlot = null;
front.plots.nframes = null;

var dragFunctions = {
    
    // methods for dragging the regions in the left svg
    
    drag: d3.behavior.drag()
            .origin(function() { 
                var t  = d3.select(this);
                return {x: t.attr("x"),y: t.attr("y")};})
            .on("drag",   function()  {
                if (Object.keys(front.backendTasks).length === 0) {
                    var t = d3.select(this);
                    dragFunctions.updateBoxes(t.attr("regionId"),t.attr("location"))
		    }
                })
            .on("dragend",function () {
                // when dragging has finished, update the region information both here
                // and in the python backend. then recalculate and replot.
                if (Object.keys(front.backendTasks).length === 0) {

		    // update the region information in the tracker
		    var id   = d3.select(this).attr("regionId");
		    var group = d3.select("#regionGroup"+id)
		    
		    r = front.intensity.regions[id];
		    r.coords.rmin = parseInt(group.select('[location="mainRegion"]').attr("y"));
		    r.coords.cmin = parseInt(group.select('[location="mainRegion"]').attr("x"));
		    r.coords.rmax = parseInt(group.select('[location="lowerRight"]').attr("y"));
		    r.coords.cmax = parseInt(group.select('[location="lowerRight"]').attr("x"));   
    
		    // run the recalculation if that option is selected
                    if ($('#autocalc').is(":checked")) { userFunctions.recalculateG2() };
		}
	    }),
            
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
        var ds = front.sizes.d, ss = front.sizes.s, wh = front.sizes.x, ww = front.sizes.x;
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
	    var rs = front.sizes.r, ds = front.sizes.d, ss = front.sizes.s;
	    var allBoxes = [
		{h:rs, w: rs, x:tc.cmin,    y:tc.rmin,    c:"mainRegion"},
		{h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin+rs, c:"lowerRight"},
		{h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin-ds, c:"upperRight"},
		{h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin+rs, c:"lowerLeft"},
		{h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin-ds, c:"upperLeft"}];
		
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
		    .call(dragFunctions.drag) // attaches the dragging behavior
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
	    for (region in front.intensity.regions) {hues.push(front.intensity.regions[region].hue);}
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
	    rmin:(front.sizes.x-front.sizes.r)/2,
	    rmax:(front.sizes.x+front.sizes.r)/2,
	    cmin:(front.sizes.x-front.sizes.r)/2,
	    cmax:(front.sizes.x+front.sizes.r)/2}
	
	// identifiers
	var t = new Date().getTime()
	r.regionId = t
	r.hue      = newHue()
	r.color    = d3.hsl(r.hue*360,1,0.5).toString();
	r.selected = false;
	
	// data and fit
	r.functional = null;
	r.g2Values   = [];
	r.fitValues  = [];
	r.fit        = {};
	
	// add to tracker
	front.intensity.regions[t] = r;
	
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
    
    selectRegion: function (t) {
        // given the regionId "t", select the region in the tracker
	front.intensity.regions[t].selected = !front.intensity.regions[t].selected;
	
	// select the region in the svg; switch the opacity
	var c = d3.select("#regionGroup"+t).select("circle");
        var o = 1-c.style("fill-opacity");
        c.transition().duration(150).style("fill-opacity",o)
    },
    
    changeBackground: function () {
        // the firstframe image is 9 images in 1:
	// row 1 is linear, row 2 is sqrt, row 3 is log
	// col 1 is L, col 2 is A, col 3 is B.
	// on the basis of the selected value, move the image to give the
	// appearance that the image was reloaded
        var scale = parseInt($("input[name=scale]:checked").val());
        var color = parseInt($("input[name=cm]:checked").val());
        d3.select('#dataimage').attr("x",-front.sizes.x*scale).attr("y",-front.sizes.x*color)
    },
    
    deleteRegions: function () {
        
	// delete from the backend and the frontend all the selected regions.
	// if they don't exist on the backend, it doesnt matter
        var backend = function (callback) {
            
            var ts = new Date().getTime()
            front.backendTasks[ts] = 'deleteRegions';
	    
	    $.ajax({
		url: "xpcs/remove",
		type: 'POST',
		data: JSON.stringify(selectedRegions),
		contentType: 'application/json; charset=utf-8',
		dataType: 'json',
		async: true,
		success: function(data) {
		    console.log(data);
		    delete front.backendTasks[ts]
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
                    front.plots.selectedPlot = null;
                };
            };
                
	    for (var k = 0; k < selectedRegions.length; k++) {
		thisRegion = selectedRegions[k]
		if (front.intensity.regions[thisRegion].selected) {
		    
		    // remove from dictionary
		    delete front.intensity.regions[thisRegion]
		    
		    // remove group from intensity plot
		    d3.select("#regionGroup"+thisRegion).remove()
		    
		    // remove group from g2 plot
		    d3.select("#g2Group"+thisRegion).remove()
		}
	    }
        };
	
	// find all the selected regions
	var selectedRegions = [];
	for (region in front.intensity.regions) {
	    if (front.intensity.regions[region].selected) {
		selectedRegions.push(region)
	    }
	}
	
	// remove the selected regions from the backend, then from the frontend
        queue().defer(backend).await(frontend);

    },
    
    recalculateG2: function () {

        var funcString, fileId

	// 1. send regions coordinates to the backend.
	// 2. backend calculates g2 in all regions which have changed.
	// 3. backend fits data in all regions which have changed
	// 4. backend returns as a json object the g2 and fit values
	// 5. g2 and fit get attached to front.regions
	// 6. plots get redrawn
	
	var redraw = function (error) {
	    
	    if (error != null) { console.log(error) }
	    
	    // record which line is currently selected, if applicable
	    var oldSelection = front.plots.selected;

	    // when redrawing, there is no such thing as a selected plot.
	    userFunctions.selectPlot(front.plots.selected);
	    front.plots.selected = null;

	    // the way the data is structured poses a problem for the d3 enter/exit
	    // methodology (ie, doesnt work!). instead it seems easier to simply
	    // remove all the children groups of #plotGroup and replot all data
	    svgg = d3.select('#plotGroup');
	
	    // clear all the old plots
	    svgg.selectAll(".dataSeries").remove()

	    // d3 doesn't play well with being passed an object instead of an
	    // array. therefore, recast front.intensity.regions into an array
	    // of objects with only the necessary data
	    var plottables = []
	    for (var region in front.intensity.regions) {
		var nr = {}, tr = front.intensity.regions[region]
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
		.on("click",function () {userFunctions.selectPlot(d3.select(this).attr("id"))});
	
	    newLines.append("path")
		.style("fill","none")
		.style("stroke-width",2)
		.style("stroke",function (d) {return d.color})
		.attr("class","dataPath")
		.attr("id",function(d) {return "g2Data"+d.regionId})
		.attr("d", function(d) {return front.plots.linefunc(d.g2Values); });

	    // define a scale for the size of the circles so that they decrease in
	    // size as they approach 1
	    front.plots.rScale = d3.scale.log().domain([1,front.plots.nframes]).range([7,0]).clamp(false);
	
	    // add data circles.
	    for (var k=0;k<plottables.length;k++) {
		d3.select('#g2Group'+plottables[k].regionId).selectAll(".g2circle")
		    .data(plottables[k].g2Values)
		    .enter().append("circle")
		    .attr("class","g2circle")
		    .attr("cx",function (d) {return front.plots.xScale(d.x)})
		    .attr("cy",function (d) {return front.plots.yScale(d.y)})
		    .attr("r", function (d) {return front.plots.rScale(d.x)})
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
		.attr("d", function(d) {return front.plots.linefunc(d.fitValues); })
		;   
		
	    userFunctions.selectPlot(oldSelection);
	    
	    //move the parameters box to the top
	    var g = d3.select("#fitParamsGroup")[0][0]
	    g.parentNode.appendChild(g)
	    }
	
	var parse = function (data) {
	    
	    // take the data returned from the backend fitting and attach
	    // it to the correct locations in front.intensity etc
	    var functional = data.fitting.functional;
	    var parameters = data.fitting.parameters;
	    
	    for (region in data.analysis) {
		
		thisData   = data.analysis[region];
		thisRegion = front.intensity.regions[region];
		
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
	    
	    // lock
	    var ts = new Date().getTime();
            front.backendTasks[ts] = 'recalculatePlot'
	    
	    // draw clock
	    d3.selectAll(".clockface").transition().duration(150).style("opacity",1)
	    
	    // loop over regions, building a coordinates array
	    data = {}
	    data.coords = {}
	    data.form   = $("input[name=fitform]:checked").val()
	    for (region in front.intensity.regions) {data.coords[region] = front.intensity.regions[region].coords}
	    
	    // send this array to the backend. when it comes back, parse it,
	    // undraw the clock, then draw the plots
	    $.ajax({
		    url: "xpcs/calculate",
		    type: 'POST',
		    data: JSON.stringify(data),
		    contentType: 'application/json; charset=utf-8',
		    dataType: 'json',
		    async: true,
		    success: function(data) {
			parse(data);
			delete front.backendTasks[ts]
			d3.selectAll(".clockface").transition().duration(150).style("opacity",0)
			callback(null);
			}
		});
	}
	
	queue().defer(backend).await(redraw);
    },

    selectPlot:function (groupId) {
    
        //console.log("clicked on: "+groupId)
	//console.log("current:    "+front.plots.selected)
    
        // selectors
        if (front.plots.selected != null) {
	    var oldRegionGroup = "#regionGroup"+front.plots.selected;
            var oldGroupId     = "#g2Group"+front.plots.selected;
            var oldFitId       = "#g2Fit"+front.plots.selected;}
	    
        if (groupId != null) {
            var regionId   = groupId.replace("g2Group","");
            var newGraphId = groupId;
            var newFitId   = "#g2Fit"+regionId;}
	    var newRegionGroup = '#regionGroup'+regionId
	    
        if (groupId === null) {regionId = null}

        // first, deselect the selected plot if the selected plot is not null
        if (front.plots.selected != null ) {
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
        
        var osp = front.plots.selected;
        if (osp === regionId) {front.plots.selected = null};
        if (osp  != regionId) {front.plots.selected = regionId};
        
        // update the display of the fit parameters
        if (front.plots.selected != null) {
            
	    var thisRegion = front.intensity.regions[front.plots.selected]
	    
	    // get the fit parameters
	    var fitmap = thisRegion.fitParamsMap;
	    var fitval = thisRegion.fitParamsVals;
	    var lines = [thisRegion.functional]
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
    
    initGraph: function () {
        
        // define log scales for the new data
        front.plots.xScale = d3.scale.log().range([0, front.plots.width]).domain([1,front.plots.nframes]);
        front.plots.yScale = d3.scale.log().range([front.plots.height, 0]).domain([1e-6,1]).clamp(true);
	
	// define the interpolation function for plotting
	front.plots.linefunc = d3.svg.line()
                .interpolate("linear")
                .x(function(d) { return front.plots.xScale(d.x); })
                .y(function(d) { return front.plots.yScale(d.y); });
        
        var resetSVG = function () {
	    
            d3.select("plotGroup").remove();
            d3.select("#svggraphs")
                .append("g")
                .attr("transform", "translate(" + front.plots.margins.left + "," + front.plots.margins.top + ")")
                .attr("id","plotGroup");
        };
	
        var drawGrids = function () {
            //draw grid lines
            svgg.append("g").attr("id","verticalGrid")
            d3.select("#verticalGrid").selectAll(".gridlines")
                .data(front.plots.xScale.ticks()).enter()
                .append("line")
                .attr("class","gridlines")
                .attr("x1",function (d) {return front.plots.xScale(d)})
                .attr("x2",function (d) {return front.plots.xScale(d)})
                .attr("y1",function ()  {return front.plots.yScale(1e-6)})
                .attr("y2",function ()  {return front.plots.yScale(1e0)})
    
            svgg.append("g").attr("id","horizontalGrid")
            d3.select("#horizontalGrid").selectAll(".gridlines")
                .data(front.plots.yScale.ticks()).enter()
                .append("line")
                .attr("class","gridlines")
                .attr("x1",function ()  {return front.plots.xScale(1)})
                .attr("x2",function ()  {return front.plots.xScale(front.plots.nframes)})
                .attr("y1",function (d) {return front.plots.yScale(d)})
                .attr("y2",function (d) {return front.plots.yScale(d)})
        };
                
        var drawAxes = function () {
            // draw axes
            
            // define xAxis and yAxis
            var nticks = Math.floor(Math.log(front.plots.nframes)/Math.LN10);
            var xAxis  = d3.svg.axis().scale(front.plots.xScale).orient("bottom").ticks(nticks);
            var yAxis  = d3.svg.axis().scale(front.plots.yScale).orient("left");//.ticks(5);//.tickFormat(d3.format(".1f"));
            
            svgg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + front.plots.height + ")")
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
            
        var drawClock = function () {

            // size and center
            var bw = 100, bh = 100;
            var tx = front.plots.xScale(Math.pow(10,(Math.log(front.plots.nframes)+Math.log(1))/(2*Math.LN10)));
            var ty = front.plots.yScale(Math.pow(10,(Math.log(1e-6)+Math.log(1))/(2*Math.LN10)));

            d3.select("#plotGroup")
                .append("g")
                .attr("id","clockface")
                .attr("transform","translate("+tx+","+ty+")")
                ;
                
            cf = d3.select("#clockface")
        
            cf.append("rect")
                .attr("class","clockface")
                .attr("x",-bw/2).attr("y",-bh/2)
                .attr("height",bw).attr("width",bh)
                .attr("rx",10).attr("ry",10)
                .style("stroke-width",2).style("stroke","black")
                .style("fill","red")
                ;
                
            var hands = [[{x:0,y:0},{x:0,y:30}],[{x:0,y:0},{x:30,y:0}]];
        
            var circles = [
                {cx:0,cy:0,r:bw*.4,f:"white",s:"black",sw:2},
                {cx:0,cy:0,r:5,f:"black",s:"black",sw:0}]
            
            for (var n=0;n<12;n++) {
                var nc = {f:"black",s:"black",sw:0,r:2}
                nc.cx = bw*0.35*Math.cos(Math.PI*2/12*n);
                nc.cy = bw*0.35*Math.sin(Math.PI*2/12*n);
                circles.push(nc)
            }
            
            cf.selectAll("circle").data(circles).enter()
                .append("circle")
                .attr("class","clockface")
                .attr("cx",function(d){return d.cx})
                .attr("cy",function(d){return d.cy})
                .attr("r",function(d) {return d.r})
                .style("fill",function(d) {return d.f})
                .style("stroke",function(d){return d.s})
                .style("stroke-width",function(d){return d.sw})
                
            var handLine = d3.svg.line()
                .interpolate("linear")
                .x(function(d) { return d.x; })
                .y(function(d) { return d.y; });
        
            cf.selectAll("path").data(hands).enter().append("path")
                .attr("class","clockface")
                .attr("d", function(d) {return handLine(d)})
                .style("stroke","black")
                .style("stroke-width",2);
        };
            
        //
        resetSVG();
            
        var svgg = d3.select("#plotGroup")
        
        drawGrids()
        drawAxes()
        drawReadout()
        drawClock()

        },
    textBoxTransition: function (newOpacity,regionId) {
        if (regionId === null)
            {regionId = front.plots.selected}
        if (newOpacity === 0) {
            d3.selectAll("#txtrect").transition().duration(150).style("opacity",0);
            d3.selectAll(".fitText").transition().duration(150).style("opacity",0).remove();
            d3.select("#fitParamsText").attr("selectedGroup","none");
            }
        if (newOpacity === 1) {
            var txt = d3.select("#fitParamsText");
            var bbox = txt.node().getBBox();
	    
	    var tbx = front.plots.xScale(1)+10;
	    var tby = front.plots.yScale(1e-6)-10-bbox.height;
	    
            d3.select("#fitParamsGroup").attr("transform","translate("+tbx+","+tby+")");
            d3.select("#txtrect").attr("width",bbox.width+10).attr("height",bbox.height);
            d3.selectAll(".fitText").transition().duration(150).style("opacity",1);
            d3.select("#txtrect").transition().duration(150).style("opacity",1);}
    },

    };

var start = function () {
    
    // this is code that should run when the script is done loading.

    // using jquery, attach actions to the interface buttons. the length check
    // makes sure actions cannot be executed while a backend task (such as
    // calculating g2) is currently running.
    $("#load_data").click(function() {if (Object.keys(front.backendTasks).length === 0) {userFunctions.loadData()};});
    $("#new_region").click(function() {if (Object.keys(front.backendTasks).length === 0) {userFunctions.addRegion()}});
    $("#delete").click(function() {if (Object.keys(front.backendTasks).length === 0) {userFunctions.deleteRegions()};});
    $("#recalculate").click(function() {if (Object.keys(front.backendTasks).length === 0) {userFunctions.recalculateG2()};});
    $("#colormaps").children().click(function () {userFunctions.changeBackground();});
    $("#scales").children().click(function () {userFunctions.changeBackground();});
    $("#functionals").children().click(function () {if (Object.keys(front.backendTasks).length === 0) {userFunctions.recalculatePlot()};});

    // using d3, append the right things to the divs
    d3.select("#intensity")
        .append("svg")
        .attr("id","svgintensity")
        .attr("width",front.sizes.x)
        .attr("height",front.sizes.x);
        
    d3.select('#svgintensity')
        .append("image")
        .attr("id","dataimage")
        .attr("width",3*front.sizes.x)
        .attr("height",3*front.sizes.x);
        
    // skeletonize the right svg plots
    // if a graph already exists, remove it
    d3.select('#svggraphs').remove()
    
    // margins for svggraphs
    width  = front.sizes.x - front.plots.margins.left - front.plots.margins.right,
    height = front.sizes.x - front.plots.margins.top - front.plots.margins.bottom;
         
    front.plots.width  = width
    front.plots.height = height
    
    // create the chart group.
    d3.select("#graphs").append("svg").attr("id","svggraphs")
        .attr("width",  width + front.plots.margins.left + front.plots.margins.right)
        .attr("height", height + front.plots.margins.top + front.plots.margins.bottom)
	
    // query the backend to get the dataid and the number of frames
    
    var backend = function (callback) {
    
	var got = 0;
    
	$.getJSON(
            'xpcs/purge', {},
            function(returned) {
		got += 1;
		if (got === 2) {callback(null)};
                }
            );
    
	$.getJSON(
            'xpcs/query', {},
            function(returned) {
                front.intensity.dataId  = returned.dataId;
		front.plots.nframes = returned.nframes;
		got += 1;
		if (got === 2) { callback(null) };
                }
            );
        };
	
    var frontend = function (error) {
        
	var img = new Image()
	img.onload = function () {
	    d3.select('#dataimage').attr("xlink:href", path);
	    userFunctions.changeBackground();
	    userFunctions.initGraph();
	}
	
	var path = '/static/xpcs/images/data_'+front.intensity.dataId+'.jpg';
        img.src  = path
    }
    
    queue().defer(backend).await(frontend);

};

start();