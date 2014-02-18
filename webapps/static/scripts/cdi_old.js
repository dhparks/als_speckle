backendTasks = {}
gui = {}
gui.sizes = {x:300, b:20}
gui.hologram = {hasData:false,regions:{},hasSupport:false,zoom:null,zoomFactor:1.25}
gui.reconstruction = {selectedRound:null,rScale:'linr',regions:{},round:0,hasPropagationRegion:false}
gui.rftf = {rftfs:{}, margins:{top: 15, right: 15, bottom: 30, left: 40}}
gui.acutance = {exists:false,bp_strip:null}
gui.isLocked = false;

gui.hologram.translateX = 0;
gui.hologram.translateY = 0;
gui.hologram.staticTranslateX = 0;
gui.hologram.staticTranslateY = 0;

var dragFunctions = {
    
    // functions which control the dragging behavior of the region boxes
    scrubAcutance: function () {

	// get the old position
	var m    = d3.select("#acutanceMarker");
	var oldx = parseInt(m.attr("cx"));

	// transform into new location given d3.event.x
	var newx = oldx+d3.event.dx;
	var z    = gui.acutance.xScale.invert(newx)
	var idx  = Math.floor(z-gui.acutance.zmin)
	var newy = gui.acutance.yScale(gui.acutance.data[idx].y)
	
	// draw at new location
	d3.select("#acutanceMarker").attr("cx",newx).attr("cy",newy)
	
	// update the axis connecting lines
	var connect = [[{x:gui.acutance.zmin,y:gui.acutance.data[idx].y},
		        {x:gui.acutance.zmax,y:gui.acutance.data[idx].y}],
		       [{x:z,y:0},
		        {x:z,y:1}]
		       ];

	var x = d3.select("#acutancePlot").selectAll("#connect").data(connect)
	    
	// when the line is new, set its attributes
	x.enter().append("path")
	    .attr("fill","none")
	    .attr("stroke","red")
	    .attr("stroke-width",1)
	    .attr("id","connect")
	    .attr("stroke-dasharray","5,5")
	x.attr("d",function (d) {return gui.acutance.Line(d)})
	
	//change the background image
	var ix = idx%gui.acutance.g, iy = Math.floor(idx/gui.acutance.g)
	$("#propagationimage").attr('x',-gui.sizes.x*ix).attr('y',-gui.sizes.x*iy);
    },
    
    updateBoxes: function (who,where,what) {

	// this function updates the coordinates of the 6 svg elements which
	// show a region. most of the mathematical complexity arises from the
	// decision to enforce boundary conditions at the edge of the data, so
	// no aspect of the region becomes undraggable.

	var group = d3.select('#imageRegion_'+who);

	var mr = group.select('[location="mainRegion"]');
	var ul = group.select('[location="upperLeft"]');
	var ur = group.select('[location="upperRight"]');
	var lr = group.select('[location="lowerRight"]');
	var ll = group.select('[location="lowerLeft"]');
	var cs = group.select('[location="selecter"]');
	var cw = parseInt(mr.attr("width"));
	var ch = parseInt(mr.attr("height"));
	
	// define shorthands so that lines don't get out of control
	var mx, my, ulx, uly, urx, ury, llx, lly, lrx, lry, csx, csy;;
	var ds = 5, wh = gui.sizes.x, ww = gui.sizes.x, ss = 5;
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
	    if (what === 'lowerLeft')  {new_x1 = x1b; new_x2 = x2; new_y1 = y1;  new_y2 = y2b;}
	    if (what === 'lowerRight') {new_x1 = x1; new_x2 = x2b; new_y1 = y1;  new_y2 = y2b;}
	    var new_width  = new_x2-new_x1;
	    var new_height = new_y2-new_y1;
	    
	    // assign the coordinates
	    mrx = new_x1+ds; mry = new_y1+ds;
	    ulx = new_x1;    uly = new_y1;
	    urx = new_x2;    ury = new_y1;
	    llx = new_x1;    lly = new_y2;
	    lrx = new_x2;    lry = new_y2;
	    csx = new_x1+(new_width+ds)/2; csy = new_y1;
	    
	    mr.attr("width",new_width-ds).attr("height",new_height-ds)    
	}
    
	// update the positions
	mr.attr("x",mrx).attr("y",mry)
	ul.attr("x",ulx).attr("y",uly);
	ur.attr("x",urx).attr("y",ury);
	ll.attr("x",llx).attr("y",lly);
	lr.attr("x",lrx).attr("y",lry);
	cs.attr("cx",csx).attr("cy",csy);
    
    },

    updateRegion: function (who,where) {
	
	var w2 = where.replace("svg","")

	// update the region as known by javascript
	var group = d3.select("#imageRegion_"+who);
	console.log(who,where)
	
	var region = gui[w2].regions[who];
	
	console.log(region)

	region.coords.rmin = parseInt(group.select('[location="mainRegion"]').attr("y"));
	region.coords.cmin = parseInt(group.select('[location="mainRegion"]').attr("x"));
	region.coords.rmax = parseInt(group.select('[location="lowerRight"]').attr("y"));
	region.coords.cmax = parseInt(group.select('[location="lowerRight"]').attr("x"));   
	},
	
    updateBackground: function () {

	// update the gui.translateX and gui.translateY parameters
	// with the information coming from the drag event; then
	// update the image transform properties
	
	var dx = d3.event.x, dy = d3.event.y;
	gui.hologram.translateX = gui.hologram.staticTranslateX+dx/gui.hologram.zoom;
	gui.hologram.translateY = gui.hologram.staticTranslateY+dy/gui.hologram.zoom;
	var str = "scale("+gui.hologram.zoom+") translate("+gui.hologram.translateX+","+gui.hologram.translateY+")"
	d3.select('#hologramImage').attr("transform",str)
	
    },	
    
    // this is the drag behavior for the hologram image
    imgDragBehavior: d3.behavior.drag()
	.origin(function() { 
	    var t  = d3.select(this);
	    return {x: t.attr("x"),y: t.attr("y")};})
	.on("drag", function() {
		if (!gui.isLocked) {
		    console.log('drag2')
		    var t = d3.select(this);
		    dragFunctions.updateBackground()}
		})
	.on("dragend",function () {
	    // when dragging has finished, update the javascript information about the position
	    if (!gui.isLocked) {
		gui.hologram.staticTranslateX = gui.hologram.translateX;
		gui.hologram.staticTranslateY = gui.hologram.translateY;
	    }
	}),
    
	
};

var userFunctions = {
    
    buttonClick: function (what, who) {
	
	console.log("switchboard to "+what)
	
	// switching function to handle button-click actions
	if (what === "zoomIn") {userFunctions.zoomIn()}
	if (what === "zoomOut") {userFunctions.zoomOut()}
	if (what === "newRegion") {
	    if (who==="svghologram" || (who==="svgreconstruction" && gui.reconstruction.regionId===null)) {
		userFunctions.newRegion(who)}}	
	if (what === "delRegion") {userFunctions.delRegion()}
	if (what === "reconstruct") {userFunctions.reconstruct(false)}
	if (what === "reconstructAgain") {userFunctions.reconstruct(true)}
	if (what === "breadCrumb") {userFunctions.presentNewData(who)}
	if (what === "propagate") {userFunctions.propagate()}
	if (what === "download") {userFunctions.download()}
    },
    
    download: function () {
	
	var backend  = function (callback) {
	    // send a command to the imaging backend to save the current
	    // selection to a zipfile on disk
	    
	    $.getJSON(
		'cdi/download',
		{'reconstructionId':gui.reconstruction.selectedRound},
		function (results) { console.log(results); callback(null)}
	    );
	};
	
	var frontend = function (error) {
	    
	    if (error != null) {
		console.log("ajax error in download")
		console.log(error)
	    };
	    
	    console.log("here1")
	    
	    var name = "reconstruction_id"+gui.dataId+"_round"+gui.reconstruction.selectedRound+"_zipped.gz"
	    var fileId = "static/imaging/fits/"+name;
	    
	    var save = document.createElement('a');
	    save.href = fileId;
	    save.target = '_blank';
	    save.download = name;
	    
	    console.log("here2")
	    console.log(save)

	    var event = document.createEvent('Event');
	    event.initEvent('click', true, true);
	    save.dispatchEvent(event);
	    (window.URL || window.webkitURL).revokeObjectURL(save.href)
	    
	    console.log("here3")
	    
	    console.log(window.URL)
	    
	}
	
	queue().defer(backend).await(frontend)
	
    },
    
    propagate: function () {

	var resetSVG = function () {

	    d3.select("#acutanceGroup").remove();
	    d3.select("#svgacutance")
		.append("g")
    		.attr("transform", "translate(" + gui.acutance.margins.left + "," + gui.acutance.margins.top + ")")
		.attr("id","acutanceGroup");
		};
		
	var drawGrids = function () {
		    
	    svga = d3.select("#acutanceGroup")
	    svga.append("g").attr("id","verticalGrid")
	    svga.append("g").attr("id","horizontalGrid")

	    x = svga.select("#verticalGrid").selectAll(".gridlines").data(gui.acutance.xScale.ticks()).enter()
	    y = svga.select("#horizontalGrid").selectAll(".gridlines").data(gui.acutance.yScale.ticks(5)).enter()

	    x.append("line")
		.attr("class","gridlines")
		.attr("x1",function (d) {return gui.acutance.xScale(d)})
		.attr("x2",function (d) {return gui.acutance.xScale(d)})
		.attr("y1",function ()  {return gui.acutance.yScale(0)})
		.attr("y2",function ()  {return gui.acutance.yScale(1)})

	    y.append("line")
		.attr("class","gridlines")
		.attr("x1",function ()  {return gui.acutance.xScale(gui.acutance.zmin)})
		.attr("x2",function ()  {return gui.acutance.xScale(gui.acutance.zmax)})
		.attr("y1",function (d) {return gui.acutance.yScale(d)})
		.attr("y2",function (d) {return gui.acutance.yScale(d)})
	    };
	    
	var drawAxes = function () {
	    // draw axes
    
	    svga = d3.select("#acutanceGroup")
    
	    // define xAxis and yAxis
	    var xAxis  = d3.svg.axis().scale(gui.acutance.xScale).orient("bottom");
	    var yAxis  = d3.svg.axis().scale(gui.acutance.yScale).orient("left").ticks(5);
    
	    svga.append("g")
		.attr("class", "x plotaxis")
		.attr("transform", "translate(0," + gui.acutance.height + ")")
		.call(xAxis)
		.append("text")
		.attr("x", gui.acutance.width-10)
		.attr("dy", "-0.4em")
		.style("text-anchor", "end")
		.text("z (um)");
	  
	    svga.append("g")
		.attr("class", "y plotaxis")
		.attr("transform","translate(0,0)")
		.call(yAxis)
		.append("text")
		.attr("transform", "rotate(-90)")
		.attr("y",6)
		.attr("dy", ".7em")
		.style("text-anchor", "end")
		.text("Accutance");
	};
	
	var drawPlot = function () {
	    
	    svga = d3.select('#acutanceGroup');
	
	    // clear the old plot
	    svga.select("#acutancePlot").remove()
	    svga.append("g").attr("id","acutancePlot")

	    // make the new plot
	    d3.select("#acutancePlot").selectAll("path")
		.data([gui.acutance.data])
		.enter()
		.append("path")
		.attr("d",function (d) {return gui.acutance.Line(d)})
		.attr("fill","none")
		.attr("stroke","black")
		.attr("stroke-width",2);
		
	    var z0 = gui.acutance.xScale(0)
	    var a0 = gui.acutance.yScale(gui.acutance.data[Math.abs(gui.acutance.zmin)].y)
	    
	    var x0 = 0
	    var y0 = gui.acutance.data[Math.abs(gui.acutance.zmin)].y
	    
	    var connect = [
		[{x:gui.acutance.zmin,y:y0},
		 {x:gui.acutance.zmax, y:y0}],
		[{x:x0,y:0},
		 {x:x0,y:1}]
		];

	    var x = d3.select("#acutancePlot").selectAll("#connect").data(connect)
	    
	    // when the line is new, set its attributes
	    x.enter().append("path")
		.attr("fill","none")
		.attr("stroke","red")
		.attr("stroke-width",1)
		.attr("id","connect")
		.attr("stroke-dasharray","5,5")
		.attr("d",function (d) {return gui.acutance.Line(d)})
		
	     // draw the marker ball
	    d3.select("#acutancePlot").append("circle")
		.attr("cx", z0)
		.attr("cy", a0)
		.attr("r",5)
		.attr("fill","red")
		.attr("id","acutanceMarker")
		.style("cursor","move");
		
	    var drag = d3.behavior.drag()
		.origin(function() { 
		    var t  = d3.select(this);
		    return {x: t.attr("x"),y: t.attr("y")};})
		.on("drag", function() {
		    var t = d3.select(this);
		    dragFunctions.scrubAcutance()})

	    // attach the dragging behavior
	    d3.select("#acutanceMarker").call(drag);
		
		
	}
		
	var backend = function (callback) {
		    
	    // send the region coordinates to the backend along with
	    // energy etc. from this, the backend will calculate
	    // the pixel coordinates of the region and do the back
	    // propagation
	    
	    var coords = gui.reconstruction.regions[gui.reconstruction.regionId].coords
	    var ap = 1;
	    
	    // format the request
	    var url  = "cdi/propagate"
	    var info = {'rmin':1.*coords.rmin/gui.sizes.x,
			'rmax':1.*coords.rmax/gui.sizes.x,
			'cmin':1.*coords.cmin/gui.sizes.x,
			'cmax':1.*coords.cmax/gui.sizes.x,
			'round':gui.reconstruction.selectedRound,
			'energy':e,'zmin':z1,'zmax':z2,'pitch':p,'apodize':ap};
	    
	    $.getJSON(url, info,
		function(json_data) {
		    console.log(json_data.result)
		    gui.acutance.propagationId = json_data.propagationId
		    
		    // load the propagation sprite. set the background correctly.
		    gui.acutance.bp_strip = new Image();
		    gui.acutance.bp_strip.onload = function () {
			var w = this.width, h = this.height;
			gui.acutance.g = w/gui.sizes.x;
			
			var idx = Math.abs(gui.acutance.zmin)
			var ix = idx%gui.acutance.g, iy = Math.floor(idx/gui.acutance.g)
			$("#propagationimage").attr('x',-gui.sizes.x*ix).attr('y',-gui.sizes.x*iy);
			
			d3.select("#propagationimage")
			    .attr("width",w)
			    .attr("height",h)
			    .attr("x",-gui.sizes.x*ix)
			    .attr("y",-gui.sizes.x*iy)
			    .attr("xlink:href",gui.acutance.bp_strip.src);
			userFunctions.unlock();
			callback(null)
			};
		
		    var path = 'static/imaging/images/bp_session'+gui.sessionId+'_id'+gui.acutance.propagationId+'.jpg'
		    gui.acutance.bp_strip.src = path;
		    
		}
	    );
	};
	
	var frontend = function (error) {
	    
	    var afterLoad = function (error,data) {
		
		// parse the data
		gui.acutance.data = data.map(function (d) {return {x:parseFloat(d.z),y:parseFloat(d.acutance)}})
		
		// define new scales
		gui.acutance.xScale = d3.scale.linear().range([0, gui.acutance.width]).domain([gui.acutance.zmin,gui.acutance.zmax]);
		gui.acutance.yScale = d3.scale.linear().range([gui.acutance.height, 0]).domain([0,1]);
		
		// now draw the plot; component functions are broken up for readability
		resetSVG();
		drawAxes();
		drawGrids();
		drawPlot();
		
		// bring up the opacity of the group
		d3.select("#acutance").transition().duration(250).style("opacity",1);
		
	    };
	    
	    queue()
		.defer(d3.csv, '/static/imaging/csv/acutance_session'+gui.sessionId+'_id'+gui.acutance.propagationId+'.csv')
	    	.await(afterLoad);
	};

	var e = $('#energy').val(), p = $('#pitch').val(), z1 = $('#zmin').val(), z2 = $('#zmax').val();
	
	var typesOK = !(isNaN(parseFloat(e)) || isNaN(parseFloat(p)) || isNaN(parseFloat(z1)) || isNaN(parseFloat(z2)))
	
	// save zmin and zmax to the front tracker. if zmin > zmax, reverse the assignment
	var z3 = parseInt(z1);
	var z4 = parseInt(z2);
	if (z3 > z4) {gui.acutance.zmin = z4; gui.acutance.zmax = z3};
	if (z4 > z3) {gui.acutance.zmin = z3; gui.acutance.zmax = z4};
	if (z4 === z3) {typesOK = false};

    	if (e != '' && p != '' && z1 != '' && z2 != '' && typesOK) {
	    userFunctions.lock()
	    queue().defer(backend).await(frontend)};

    },
    
    presentNewData: function (who) {
	
	who = parseInt(who)
	
	var t = d3.select("#breadcrumb"+who);
	if (gui.reconstruction.selectedRound != null) {var o = d3.select("#breadcrumb"+gui.reconstruction.selectedRound)};
	var newScale, newColor
	
	var sqrtColor = "cyan";
	
	// if "who" is not the selected crumb, select the current crumb
	// and deselect the old crumb. draw the rftf for the round.
	if (who != gui.reconstruction.selectedRound) {

	    // breadcrumbs
	    if (gui.reconstruction.selectedRound != null) { 
		o.transition().duration(150).attr("r",3).style("fill","white");}
	    if (gui.reconstruction.rScale === "linr") {newColor = "white"}
	    if (gui.reconstruction.rScale === "sqrt") {newColor = sqrtColor}
	    t.transition().duration(150).attr("r",5).style("fill",newColor);
	    
	    //rftfs
	    
	    // clear the old plot
	    svgr = d3.select('#rftfGroup');
	    svgr.select("#rftfPlot").remove();
	    
	    // draw the new plot
	    svgr.append("g").attr("id","rftfPlot")
	    var rftfData = gui.rftf.rftfs[who]
	    d3.select("#rftfPlot").selectAll("path")
		.data([rftfData,])
		.enter()
		.append("path")
		.attr("d",function (d) {return gui.rftf.Line(d)})
		.attr("fill","none")
		.attr("stroke","black")
		.attr("stroke-width",2);

	    var newScale = gui.reconstruction.rScale;
	}
    
	// if "who" is the selected crumb, maintain the size of the current
	// crumb, but switch its color and switch the scaling of the background.
	// no action is taken on the rftf plot
	if (who === gui.reconstruction.selectedRound) {
	    if (gui.reconstruction.rScale === "linr") {newColor = sqrtColor; newScale = "sqrt"}
	    if (gui.reconstruction.rScale === "sqrt") {newColor = "white"; newScale = "linr"}
	    o.transition().duration(150).style("fill",newColor);
	    gui.reconstruction.rScale = newScale;
	}
	    
	// set the new background
	var path = 'static/imaging/images/r_session'+gui.sessionId+'_id'+who+'_'+newScale+'.png'
	d3.select("#reconstructionimage").attr("xlink:href",path);
	
	gui.reconstruction.selectedRound = who;
	
    },
    
    zoomIn: function () {userFunctions.zoom(gui.hologram.zoomFactor)},
    
    zoomOut: function () {userFunctions.zoom(1./gui.hologram.zoomFactor)},

    zoom: function(factor) {
	
	// the current center coordinates  given by:
	var cc = userFunctions.convertCoord(gui.sizes.x/2);

	// update the zoom factor
	gui.hologram.zoom *= factor;
	    
	// calculate the new translation values to keep the current center
	gui.hologram.staticTranslateX = gui.sizes.x/(2*gui.hologram.zoom)-cc.x;
	gui.hologram.staticTranslateY = gui.sizes.x/(2*gui.hologram.zoom)-cc.y;
	    
	// apply the updated transformation
	var str = "scale("+gui.hologram.zoom+") translate("+gui.hologram.staticTranslateX+","+gui.hologram.staticTranslateY+")"
	d3.select('#hologramimage').attr("transform",str);

    },
    
    convertCoord: function (val) {
	return {'x':(val-gui.hologram.staticTranslateX*gui.hologram.zoom)/gui.hologram.zoom,
		'y':(val-gui.hologram.staticTranslateY*gui.hologram.zoom)/gui.hologram.zoom}
    },
	
    newRegion: function (where) {

	// default is to make regions on the hologram, but 1 region is
	// make on the reconstruction to select back-propagation
	if (typeof(where) === 'undefined') {where = "svghologram"}
	
	// draw a draggable region. coordinates are relative to the svg box, and
	// must be transformed by the backend to bring them into accordance with
	// the current zoom level.
	var region = {}
	//var rs = gui.regionSize, ds = gui.dragSize, ss = gui.selectSize, tlds=gui.tlds;
	var rs = 30, ds = 5, ss = 5, fx=gui.sizes.x;
	var rc = {'rmin':fx/2-rs/2,'rmax':fx/2+rs/2,'cmin':fx/2-rs/2,'cmax':fx/2+rs/2};
	
	// identify the region and add it to the frontend tracker
	var uid = new Date().getTime()+''
	region['uid'] = uid;
	region['coords'] = rc;
	region['selected'] = false;
	
	// note which svg elements now have draggable regions
	if (where === "svghologram") {
	    gui.hologram.hasSupport=true;
	    gui.hologram.regions[uid]=region;}
	if (where === "svgreconstruction") {
	    gui.reconstruction.regions[uid]=region;
	    gui.reconstruction.regionId = uid;
	};

	// define the relevant common attributes of the boxes
	d3.select("#"+where).append("g").attr("id","imageRegion_"+uid)
	var allBoxes = [
	    {h:rs, w: rs, x:rc.cmin,    y:rc.rmin,    c:"mainRegion", curs:"move"},
	    {h:ds, w: ds, x:rc.cmin+rs, y:rc.rmin+rs, c:"lowerRight", curs:"se-resize"},
	    {h:ds, w: ds, x:rc.cmin+rs, y:rc.rmin-ds, c:"upperRight", curs:"ne-resize"},
	    {h:ds, w: ds, x:rc.cmin-ds, y:rc.rmin+rs, c:"lowerLeft",  curs:"sw-resize"},
	    {h:ds, w: ds, x:rc.cmin-ds, y:rc.rmin-ds, c:"upperLeft",  curs:"nw-resize"}];

	var group = d3.select("#imageRegion_"+uid)
    
	// make the rectangular elements using d3
	for (var k=0;k<allBoxes.length;k++){
    
	    var thisBox = allBoxes[k];
	    var newBox  = group.append("rect")
    
	    newBox
		.attr("x",thisBox.x)
		.attr("y",thisBox.y)
		.attr("height",thisBox.h)
		.attr("width",thisBox.w)
		.attr("location",thisBox.c)
		.attr("uid",uid)
		.attr("where",where)
		.style("cursor",thisBox.curs)
		.style("fill","white")
		.style("fill-opacity",0)
		
	    if (thisBox.c==="mainRegion") {
		newBox.style("stroke","white")
		.style("stroke-width",2)
		.style("stroke-opacity",1);}
	    if (thisBox.c !="mainRegion") {
		newBox.classed("dragger",true)
		.style("fill-opacity",1);}
		
	    var drag = d3.behavior.drag()
		.origin(function() { 
		    var t  = d3.select(this);
		    return {x: t.attr("x"),y: t.attr("y")};})
		.on("drag", function() {
		    if (Object.keys(backendTasks).length === 0) {
			var t = d3.select(this);
			dragFunctions.updateBoxes(t.attr("uid"),t.attr("where"),t.attr("location"))}
		    })
		.on("dragend",function () {
		    // when dragging has finished, update the region information both here
		    // and in the python backend. then recalculate and replot.
		    if (Object.keys(backendTasks).length === 0) {
			var t = d3.select(this);
			dragFunctions.updateRegion(t.attr("uid"),t.attr("where"));
			}
		});

	    // attach the dragging behavior
	    newBox.call(drag);
	}

	// draw the selector
	// make the circular element
	if (where === "svghologram") {
	    group.append("circle")
		.attr("cx",rc.cmin+rs/2)
		.attr("cy",rc.rmin-ss)
		.attr("r",ss)
		.style("fill","white")
		.style("fill-opacity",0) // need a fill to toggle on clickSelect
		.style("stroke-width",2)
		.style("stroke","white")
		.attr("location","selecter")
		.attr("uid",uid)
		.style("cursor","pointer")
		.on("click",function () {
		    var t = d3.select(this);
		    var u = t.attr("uid");
		    var v = gui.hologram.regions[u].selected;
		    if (!v) {t.style("fill-opacity",1)}
		    if (v)  {t.style("fill-opacity",0)}
		    gui.hologram.regions[u].selected = !v;
		});
	}
    },
    
    delRegion: function () {
	
	// loop over all the regions in gui.regions. if they are selected,
	// remove them from gui.regions and delete their graphical
	// representation
	var rKeys = Object.keys(gui.hologram.regions)
	for (var k=0;k<rKeys.length;k++) {
	    var rk = rKeys[k]
	    var region = gui.hologram.regions[rk]
	    if (region.selected) {
		d3.select("#imageRegion_"+rk).remove();
		delete gui.hologram.regions[rk];
	    }
	}
	
	if (Object.keys(gui.hologram.regions).length === 0) {
	    gui.hologram.hasSupport = false;
	}
    },
    
    reconstruct: function (skip) {
    
	var _convertRegion = function (r) {
	    console.log(r)
	    out = {}
	    out.rmin = userFunctions.convertCoord(r.rmin).y;
	    out.rmax = userFunctions.convertCoord(r.rmax).y;
	    out.cmin = userFunctions.convertCoord(r.cmin).x;
	    out.cmax = userFunctions.convertCoord(r.cmax).x;
	    return out
	}
    
	var sendSupport = function (callback) {

	    // reset the master round counter
	    gui.reconstruction.round = 0;
	
	    // pull out the coordinates
	    regions = {}
	    var rKeys = Object.keys(gui.hologram.regions)
	    for (var k=0;k<rKeys.length;k++) {
		var rk = rKeys[k]
		var region = gui.hologram.regions[rk]
		regions[k] = _convertRegion(region.coords)
	    }
	    
	    var x = JSON.stringify(regions)
	    console.log(x)

	    // send to the backend. get json back.
	    $.ajax({
		url: "cdi/makesupport",
		type: 'POST',
		data: JSON.stringify(regions),
		contentType: 'application/json; charset=utf-8',
		dataType: 'json',
		async: true,
		success: function(data) {
		    callback(null);
		    }
	    });
	}
	
	var runRounds = function (error) {
	    
	    console.log("runRounds!")
	    
	    var currentRound = 0;
	    
	    var newBreadCrumb = function () {
		
		// see which breadcrumb is currently selected. if the last
		// breadcrumb is currently selected, add the new crumb
		// and select it. if something other than the last is selected,
		// add the new crumb but do not select it. if round == 0,
		// add the new crumb and select it.
		
		var addCrumb = function () {
		    // add a new breadcrumb
		    d3.select("#svgreconstruction").append("circle")
			.attr("r",3)
			.attr("reconstruction",gui.reconstruction.rId)
			.attr("cy",10)
			.attr("cx",10*(gui.reconstruction.round+1))
			.attr("class","breadcrumb")
			.attr("id","breadcrumb"+gui.reconstruction.rId)
			.attr("reconstruction",gui.reconstruction.rId)
			.on("click",function () {userFunctions.buttonClick("breadCrumb",d3.select(this).attr("reconstruction"))});
		}
		
		// get the number of the last added reconstruction
		var lastName
		var bc = d3.selectAll(".breadcrumb");
		var L   = bc[0].length;
		if (L === 0) {lastName = null}
		else         {lastName = bc[0][L-1].getAttribute("reconstruction")};

		addCrumb();
		if (currentRound === 0 || (currentRound > 0 && parseInt(lastName) === gui.reconstruction.selectedRound)) {
		    userFunctions.buttonClick("breadCrumb",gui.reconstruction.rId)
		}
	    };
	    
	    var success = function (data) {
		
		// on success, add a breadcrumb (maybe switch the image)
		// and update the rftf plot
		console.log("finished round "+gui.reconstruction.round)
		gui.reconstruction.rId = data.rId;
		
		// add the rftf to the frontend tracker
		rftf = []
		var l = data.rftf.length;
		var l2 = 1./l;
		for (var k=0;k<l;k++) {rftf.push({x:k*l2,y:data.rftf[k]})}
		gui.rftf.rftfs[gui.reconstruction.rId] = rftf;
		
		// update the images. when they get pulled from the server,
		// update and launch the next round.
		var path = 'static/imaging/images/r_session'+gui.sessionId+'_id'+gui.reconstruction.rId+'_linr.png'
		console.log(path);
		var img1 = new Image(), img2 = new Image();
		var loaded1 = false; loaded2 = false;
		
		var proceed = function () {
		    newBreadCrumb();
		    currentRound += 1;
		    gui.reconstruction.round += 1;
		    if (currentRound < r) { doRound(); }
		    else {userFunctions.unlock()}
		}
		
		img1.onload = function () {loaded1 = true; if (loaded2) {proceed()}};
		img2.onload = function () {loaded2 = true; if (loaded1) {proceed()}};
		
		img1.src = path;
		img2.src = path.replace("linr","sqrt");

	    };
	    
	    var doRound = function () {
		var url = "cdi/reconstruct"
		$.getJSON(url, params, success)
	    };

	    var i = parseInt($('#iterations').val());
	    var r = parseInt($('#rounds').val());
	    var n = parseInt($('#numtrials').val());
	    var s = parseFloat($('#swblur').val());
	    var t = parseFloat($('#swthreshold').val());

	    // enforce defaults not displayed in html boxes
	    params = {}
	    params.iterations = (isNaN(i))?100:i;
	    params.numtrials  = (isNaN(n))?2:n;
	    params.sw_sigma   = (isNaN(s))?2.0:s;
	    params.sw_cutoff  = (isNaN(t))?0.12:t;
	    params.rounds     = (isNaN(r))?1:r;
	    
	    // check the types of all params
	    var typesOK = !(isNaN(params.iterations) || isNaN(params.rounds) || isNaN(params.numtrials) || isNaN(params.sw_sigma) || isNaN(params.sw_cutoff));
	    console.log(params)
	    
	    if (gui.hologram.hasSupport && typesOK) {
		
		console.log("here")

		if (currentRound === 0) {userFunctions.lock();} // unlock at end of runRounds
		doRound();
		}
	}
	
	// reconstruct can be called from two different buttons; the first
	// prepends the reconstruct action with a call to backend.makesupport,
	// which resets the reconsturction. the second runs the reconstruction
	// directly, which simply continues the reconstruction

	if (gui.hologram.hasSupport) {
	    if (skip) {runRounds()}
	    else {queue().defer(sendSupport).await(runRounds)}};
    },
    
    lock: function () {
	
	gui.locked = true;
	console.log("locking")
	d3.select("#reconstructionButtons").transition().duration(250).style("visibility","hidden");
	d3.select("#hologramButtons").transition().duration(250).style("visibility","hidden");
    },
    
    unlock: function () {
	console.log("unlocking")
	d3.select("#reconstructionButtons").transition().duration(250).style("visibility","visible");
	d3.select("#hologramButtons").transition().duration(250).style("visibility","visible");
	gui.locked = false;  
    },
    
    shiftContainer: function () {
	var nDivs = $("#topdivs div").length;
	if (nDivs > 1) {
	    var width = (gui.sizes.x+4)*nDivs+(nDivs-1)*2;
	    d3.select("#container").transition().duration(150).style("margin-left",-width/2)
	}
    },
    
    validateField: function (id) {
	var who  = document.getElementById(id);
	var what = who.value;
	if (isNaN(what))  {who.className="fieldr"};
	if (!isNaN(what)) {who.className="fieldg"};
	if (what==='')    {who.className="field0"};
	}, 
}

var starts = {
    // functions for starting up DOM elements as they come online
    // as the analysis progresses

    backend: function (callback) {
	    
	// query the backend and get the dataid and the number of zoom levels
	$.getJSON("cdi/query", {},
	    function (data) {
		
		console.log(data);
		
		// pull data out of the returned json
		gui.havegpu          = data.hasgpu;
		gui.sessionId        = data.sessionId
		gui.dataId           = data.dataId;
		gui.hologram.size    = data.size;
		gui.hologram.zoom    = gui.sizes.x/gui.hologram.size;
		gui.hologram.hasData = true;

		
		// load the image off the server. given its size and the size
		// of the display window, set the zoom factor and translation
		// coordinates correctly
		var path = 'static/imaging/images/ifth_session'+gui.sessionId+'_id'+gui.dataId+'_0.8_logd.jpg';
		gui.hologram.image        = new Image()
		gui.hologram.image.onload = function () {callback(null)}
		gui.hologram.image.src    = path;
		
		console.log(path)
	    }
	)},
	
    frontend: function (error) {
	
	if (error != null) {
	    console.log(error)
	}
	
	d3.select("#container").style("margin","0 0 0 -"+(gui.sizes.x*3/2+2*4)+"px")
	starts.hologram();
	starts.reconstruction();
	starts.propagation();
	starts.rftf(1.5*gui.sizes.x,gui.sizes.x);
	starts.acutance();
    },

    propagation: function () {
	
	// add the top controls
	starts.controls('propagation')
	
	// add an svg with an image.
	d3.select("#propagation").append("svg")
	    .attr("id","svgpropagation")
	    .attr("width",gui.sizes.x)
	    .attr("height",gui.sizes.x)
	    
	// the width and height of this element is set only
	// when the propagation finishes
	d3.select("#svgpropagation").append("image")
	    .attr("id","propagationimage")
	
	},

    hologram: function () {
	
	fx = gui.sizes.x
	bs = gui.sizes.b
	p  = 10; // padding (when gui.sizes.x = 300)
	s  = 5;  // spacing (when gui.sizes.x = 300)

	// add the paramters controls for the reconstruction
	starts.controls('reconstruction')
	    
	// add buttons to the hologram window
	// add svgs to hologram panel
	d3.select("#hologram").append("svg")
	    .attr("id","svghologram")
	    .attr("width",gui.sizes.x)
	    .attr("height",gui.sizes.x)
	    
	d3.select("#svghologram").append("image")
	    .attr("id","hologramimage")
	    .attr("width", gui.hologram.image.width)
	    .attr("height",gui.hologram.image.height)
	    .attr('xlink:href',gui.hologram.image.src)
	    .attr('transform','scale ('+gui.hologram.zoom+')')
	    .style("cursor","crosshair")
	    .call(dragFunctions.imgDragBehavior);
	    
	d3.select("#svghologram").append("g")
	    .attr("id","hologramButtons")
	    
	// each button is defined by an object with the following parameters
	// 1. group translation (position)
	// 2. action 
	// 3. path points
	
	var bs1 = bs*0.2, bs2 = bs*0.5, bs3 = bs*0.8
	var buttons = [
	    {x:s,y:s,action:"zoomIn",paths:[[{x:bs1,y:bs2},{x:bs3,y:bs2}],[{x:bs2,y:bs1},{x:bs2,y:bs3}]]},
	    {x:bs+2*s,y:s,action:"zoomOut",paths:[[{x:bs1,y:bs2},{x:bs3,y:bs2}],]},
	    {x:fx-bs-s,y:s,action:"reconstruct",paths:[[{x:bs1,y:bs2},{x:bs3,y:bs2}],[{x:bs2,y:bs1},{x:bs3,y:bs2},{x:bs2,y:bs3}]]},
	    {x:s,y:fx-bs-s,action:"newRegion",paths:[[{x:bs1,y:bs1},{x:bs3,y:bs1},{x:bs3,y:bs3},{x:bs1,y:bs3},{x:bs1,y:bs1-1}],]},
	    {x:bs+2*s,y:fx-bs-s,action:"delRegion",paths:[[{x:bs1,y:bs1},{x:bs3,y:bs3}],[{x:bs3,y:bs1},{x:bs1,y:bs3}]]},
	];
	
	buttons.forEach(function (d) {starts.buttons(d,'#hologramButtons','hologramButtons')})
	
    },
    
    reconstruction: function () {
	
	// when the reconstruction is instantiated with a new
	// support, remove downstream analysis:
	// 1. the reconstruction image
	// 2. the propagation image
	// 3. the rftf data series
	// 4. the acutance plot

	d3.select("#svgreconstruction").remove()
	gui.reconstruction.regions = {}
	gui.reconstruction.regionId = null;
	
	d3.select("#svgacutance").remove()
	d3.select("#svgpropagation").remove()
		
	d3.select("#reconstruction").append("svg")
	    .attr("id","svgreconstruction")
	    .attr("width",gui.sizes.x)
	    .attr("height",gui.sizes.x);
	    
	d3.select("#svgreconstruction").append("image")
	    .attr("id","reconstructionimage")
	    .attr("width",gui.sizes.x)
	    .attr("height",gui.sizes.x);
	    
	// the buttons start with no opacity
	d3.select("#svgreconstruction").append("g")
	    .attr("id","reconstructionButtons")
	    .style("visibility","hidden");
	    
	var bs1 = bs*0.2, bs2 = bs*0.5, bs3 = bs*0.8
	var buttons = [
	    {x:fx-bs-s,y:s,action:"propagate",paths:[[{x:bs1,y:bs2},{x:bs3,y:bs2}],[{x:bs2,y:bs1},{x:bs3,y:bs2},{x:bs2,y:bs3}]]},
	    {x:s,y:fx-bs-s,action:"reconstructAgain",paths:[[{x:bs1,y:bs2},{x:bs3,y:bs2}],[{x:bs2,y:bs1},{x:bs2,y:bs3}]]},
	    {x:fx-2*(bs+s),y:s,action:"newRegion",paths:[[{x:bs1,y:bs1},{x:bs3,y:bs1},{x:bs3,y:bs3},{x:bs1,y:bs3},{x:bs1,y:bs1-1}]]},
	    {x:bs+2*s,y:fx-bs-s,action:"download",paths:[[{y:bs1,x:bs2},{y:bs3,x:bs2}],[{y:bs2,x:bs1},{y:bs3,x:bs2},{y:bs2,x:bs3}]]}
	];
	    
	buttons.forEach(function (d) {starts.buttons(d,'#reconstructionButtons','reconstructionButtons')});
	    
    },
    
    controls: function (forWhat) {
	
	var o = "onkeyup", f = "userFunctions.validateField(this.id)"
	var specs = {
		'reconstruction':[
		    {'type':'text','id':'rounds','placeholder':'Rounds','size':5,"onkeyup":f},
		    {'type':'text','id':'numtrials','placeholder':'Trials','size':5,"onkeyup":f},
		    {'type':'text','id':'iterations','placeholder':'Iterations','size':7,"onkeyup":f},
		    {'type':'text','id':'swblur','placeholder':'Blur (px)','size':7,"onkeyup":f},
		    {'type':'text','id':'swthreshold','placeholder':'Threshold','size':7,"onkeyup":f}],
		'propagation':[
		    {'type':'text','id':'energy','placeholder':'Energy (eV)','size':10,"onkeyup":f},
		    {'type':'text','id':'zmin','placeholder':'Zmin (um)','size':10,"onkeyup":f},
		    {'type':'text','id':'zmax','placeholder':'Zmax (um)','size':10,"onkeyup":f},
		    {'type':'text','id':'pitch','placeholder':'Pitch (nm)','size':10,"onkeyup":f},],
		'blocker':[
		    {'type':'text','id':'blockerpower','placeholder':'Power','size':10,}, 
		]
	}

	var spec = specs[forWhat]
	
	// add a div
	var x = d3.select('#controls').append("div")
		.attr("id",forWhat+"Controls")
		.attr("class","controls")
		.text("\u00a0"+forWhat+" controls")
	
	// for each element in the array, add an html element with the specified
	// attributes
	for (var k=0;k<spec.length;k++) {
	    var y = x.append("input")
	    thisInput = spec[k]
	    for (attribute in thisInput) {
		y.attr(attribute,thisInput[attribute])
	    }
	}
	
    },
    
    buttons: function (spec,where,what) {
	
	var line = lineFunc = d3.svg.line()
		.interpolate("linear")
		.x(function(d) { return d.x; })
		.y(function(d) { return d.y; });
	
	// make a group with an attached action
	var x = d3.select(where)
		.append("g")
		.attr("action",spec.action)
		.attr("transform","translate("+spec.x+","+spec.y+")")
		.on("click",function () {if (!gui.locked) {
		    var who = d3.select(this)[0][0].parentNode.parentNode.getAttribute("id");
		    userFunctions.buttonClick(d3.select(this).attr("action"),who)}});
		
	// make the rectangle button
	x.append("rect")
	    .attr("class",what)
	    .attr("height",bs)
	    .attr("width",bs)
	    .attr("rx",3)
	    .attr("ry",3);

	// make the icon paths
	x.selectAll("path").data(spec.paths).enter()
	    .append("path")
	    .attr("d",function (d) {return lineFunc(d)})
	    .style("fill","none")
	    .style("stroke","black")
	    .style("stroke-width",2);
	    
    },
    
    rftf: function (width,height) {
	
	var drawAxes = function () {
	    // draw axes
    
	    svga = d3.select("#rftfGroup")
    
	    // define xAxis and yAxis
	    var xAxis  = d3.svg.axis().scale(gui.rftf.xScale).orient("bottom");
	    var yAxis  = d3.svg.axis().scale(gui.rftf.yScale).orient("left").ticks(5);
    
	    svga.append("g")
		.attr("class", "x plotaxis")
		.attr("transform", "translate(0," + gui.rftf.height + ")")
		.call(xAxis)
		.append("text")
		.attr("x", gui.rftf.width-10)
		.attr("dy", "-0.4em")
		.style("text-anchor", "end")
		.text("|q|/|q|_max");
	  
	    svga.append("g")
		.attr("class", "y plotaxis")
		.attr("transform","translate(0,0)")
		.call(yAxis)
		.append("text")
		.attr("transform", "rotate(-90)")
		.attr("y",6)
		.attr("dy", ".7em")
		.style("text-anchor", "end")
		.text("RFTF");
	};
	
	var drawGrid = function () {
	    
	    svga = d3.select("#rftfGroup")
	    
	    //draw grid lines
	    svga.append("g").attr("id","verticalGrid")
	    d3.select("#verticalGrid").selectAll(".gridlines")
		.data(gui.rftf.xScale.ticks()).enter()
		.append("line")
		.attr("class","gridlines")
		.attr("x1",function (d) {return gui.rftf.xScale(d)})
		.attr("x2",function (d) {return gui.rftf.xScale(d)})
		.attr("y1",function ()  {return gui.rftf.yScale(0)})
		.attr("y2",function ()  {return gui.rftf.yScale(1)})
    
	    svga.append("g").attr("id","horizontalGrid")
	    d3.select("#horizontalGrid").selectAll(".gridlines")
		.data(gui.rftf.yScale.ticks(5)).enter()
		.append("line")
		.attr("class","gridlines")
		.attr("x1",function ()  {return gui.rftf.xScale(0)})
		.attr("x2",function ()  {return gui.rftf.xScale(1)})
		.attr("y1",function (d) {return gui.rftf.yScale(d)})
		.attr("y2",function (d) {return gui.rftf.yScale(d)})
	}
		
	if (typeof(width)==="undefined") {width=gui.sizes.x}
	if (typeof(height)==="undefined") {height=gui.sizes.x}
	gui.rftf.width   = width - gui.rftf.margins.left - gui.rftf.margins.right;
	gui.rftf.height  = height - gui.rftf.margins.bottom - gui.rftf.margins.top;
	
	d3.select("#rftf").append("svg")
	    .attr("id","svgrftf")
	    .attr("width",width+6)
	    .attr("height",height)
	    //.attr("visibility","hidden")
	    
	d3.select("#svgrftf")
	    .append("g")
	    .attr("transform", "translate(" + gui.rftf.margins.left + "," + gui.rftf.margins.top + ")")
	    .attr("id","rftfGroup");

	// define the scales
	gui.rftf.xScale = d3.scale.linear().range([0, gui.rftf.width]).domain([0,1]);
	gui.rftf.yScale = d3.scale.linear().range([gui.rftf.height, 0]).domain([0,1]);
	
	// line interpolator for the rftf plot
	gui.rftf.Line = d3.svg.line()
	    .interpolate("linear")
	    .x(function(d) { return gui.rftf.xScale(d.x); })
	    .y(function(d) { return gui.rftf.yScale(d.y); });
	    
	drawAxes();
	drawGrid();
    },
    
    acutance: function () {

	// acutance
	d3.select("#acutance").append("svg")
	    .attr("id","svgacutance")
	    .attr("width",1.5*gui.sizes.x)
	    .attr("height",gui.sizes.x)
	    
	gui.acutance.Line = d3.svg.line()
            .interpolate("linear")
            .x(function(d) { return gui.acutance.xScale(d.x); })
            .y(function(d) { return gui.acutance.yScale(d.y); });   
	    
	gui.acutance.exists = true;
	gui.acutance.margins = gui.rftf.margins
	gui.acutance.width   = gui.rftf.width
	gui.acutance.height  = gui.rftf.height
	

	},
 
    // query the backend, then turn on div elements
    start: function() {
	console.log("in cdi.js");
	queue().defer(starts.backend).await(starts.frontend)
    }
    
}

// run the initial start commands. other elements are started later
starts.start()