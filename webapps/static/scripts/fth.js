var backendTasks = {}
var front = {dragSize:5,regionSize:30,svgHeight:300,svgWidth:300,tlds:300,filename:null,hasdata:false}

var lock = function (id) {backendTasks[id] = null;}
var unlock = function (id) {delete backendTasks[id];}

var dragFunctions = {
    // functions which control the dragging behavior of the region boxes

    updateBoxes: function (what) {

	// this function updates the coordinates of the 6 svg elements which
	// show a region. most of the mathematical complexity arises from the
	// decision to enforce boundary conditions at the edge of the data, so
	// no aspect of the region becomes undraggable.

	var group = d3.select('#hologramRegion');

	var mr = group.select('[location="mainRegion"]');
	var ul = group.select('[location="upperLeft"]');
	var ur = group.select('[location="upperRight"]');
	var lr = group.select('[location="lowerRight"]');
	var ll = group.select('[location="lowerLeft"]');
	var cw = parseInt(mr.attr("width"));
	var ch = parseInt(mr.attr("height"));
	
	// define shorthands so that lines don't get out of control
	var mx, my, ulx, uly, urx, ury, llx, lly, lrx, lry;
	var ds = front.dragSize, wh = front.svgHeight, ww = front.svgWidth;
	var dx = d3.event.x, dy = d3.event.y
	
	// one behavior for dragging .main...
	if (what === 'mainRegion') {
    
	    mrx = dx;      mry = dy;
	    ulx = dx-ds;   uly = dy-ds;
	    llx = dx-ds;   lly = dy+ch;
	    urx = dx+cw;   ury = dy-ds;
	    lrx = dx+cw;   lry = dy+ch;
    
	    // now check their bounds
	    if (ulx < 0) {mrx = ds; ulx = 0; llx = 0; urx = cw+ds; lrx = cw+ds;};
	    if (uly < 0) {mry = ds; uly = 0; ury = 0; lly = ch+ds; lry = ch+ds;};
	    if (urx > ww-ds) {mrx = ww-cw-ds; urx = ww-ds; lrx = ww-ds; ulx = ww-cw-2*ds; llx = ww-cw-2*ds;};
	    if (lly > wh-ds) {mry = wh-ch-ds; uly = wh-ch-2*ds; ury = wh-ch-2*ds; lly = wh-ds; lry = wh-ds;};

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
	    
	    mr.attr("width",new_width-ds).attr("height",new_height-ds)    
	}
    
	// update the positions
	mr.attr("x",mrx).attr("y",mry)
	ul.attr("x",ulx).attr("y",uly);
	ur.attr("x",urx).attr("y",ury);
	ll.attr("x",llx).attr("y",lly);
	lr.attr("x",lrx).attr("y",lry);
    
    },

    updateRegion: function (callback) {

	// update the region as known by javascript
	var group = d3.select("#hologramRegion")
	front.region.coords.rmin = parseInt(group.select('[location="mainRegion"]').attr("y"));
	front.region.coords.cmin = parseInt(group.select('[location="mainRegion"]').attr("x"));
	front.region.coords.rmax = parseInt(group.select('[location="lowerRight"]').attr("y"));
	front.region.coords.cmax = parseInt(group.select('[location="lowerRight"]').attr("x"));   

	// region information only goes to python on click of propagate button
	}
}

var userFunctions = {

    validateForm: function (id) {
	
	var who  = document.getElementById(id);
	var what = who.value;
	if (isNaN(what))  {who.className="fieldr"};
	if (!isNaN(what)) {who.className="fieldg"};
	if (what==='')    {who.className="field0"};
    }, 

    holoclick: function (what) {
	
	// deal with clicks to buttons on the hologram panel. if zoomIn or
	// zoomOut, increment/decrement the zoom level and load a new
	// background image. if propagate, call the propagate routine in the backend.
	
	var zooms = ['zoomIn','zoomOut']
	if (zooms.indexOf(what) > -1) {
	    if (what === 'zoomIn') {front.zoom += 1}
	    if (what === 'zoomOut') {front.zoom -=1}
	    if (front.zoom < 0) {front.zoom = 0}
	    if (front.zoom > front.zooms) {front.zoom = front.zooms}
	    d3.select('#hologramimage').attr("x", -300*front.zoom)};
	    
	if (what === 'propagate') {
	    // primary function
	    
	    var frontSlider = function () {
		
		// (re)draw the slider
	    
		// remove old slider
		d3.select("#slidergroup").remove()
		
		// add new slider group
		d3.select("#svgslider")
		    .append("g")
		    .attr("transform", "translate(" + front.slider.margins.left + "," + front.slider.margins.top + ")")
		    .attr("id","slidergroup");
    
		// define new scale to match requested zmax
		front.slider.sx = d3.scale.linear()
		    .domain([front.zmin, front.zmax])
		    .range([0, front.slider.width])
		    .clamp(true);
		
		// define the d3 brush action
		front.slider.brush = d3.svg.brush()
		    .x(front.slider.sx)
		    .extent([0, 0])
		    .on("brush", userFunctions.brushed);
		    
		// create the x-axis
		d3.select("#slidergroup").append("g")
		    .attr("class", "x axis")
		    .attr("id","slideraxis")
		    .attr("transform", "translate(0," + front.slider.height / 2 + ")")
		    .call(d3.svg.axis()
			.scale(front.slider.sx)
			.orient("bottom")
			.tickFormat(function(d) { return d; })
			.tickSize(0)
			.tickPadding(12));
    
		// create the slider
		var slider = d3.select("#slidergroup").append("g")  
		    .attr("class", "slider")
		    .style("cursor","move")
		    .call(front.slider.brush);
		
		// unknown purpose
		slider.selectAll(".extent,.resize")
		    .remove();
		
		// unknown purpose
		slider.select(".background")
		    .attr("height", front.slider.height);
		
		// draw the handle element
		slider.append("circle")
		    .attr("class", "handle")
		    .attr("id","handle")
		    .attr("transform", "translate(0," + front.slider.height / 2 + ")")
		    .attr("r", 9);
	    }
	    
	    
	    var resetSVG = function () {
		d3.select("#acutanceGroup").remove();
		d3.select("#svgacutance")
		    .append("g")
		    .attr("transform", "translate(" + front.acutance.margins.left + "," + front.acutance.margins.top + ")")
		    .attr("id","acutanceGroup");
	    };
	    
	    var drawGrids = function () {
		
		svga = d3.select("#acutanceGroup")
		
		//draw grid lines
		svga.append("g").attr("id","verticalGrid")
		d3.select("#verticalGrid").selectAll(".gridlines")
		    .data(front.acutance.xScale.ticks()).enter()
		    .append("line")
		    .attr("class","gridlines")
		    .attr("x1",function (d) {return front.acutance.xScale(d)})
		    .attr("x2",function (d) {return front.acutance.xScale(d)})
		    .attr("y1",function ()  {return front.acutance.yScale(0)})
		    .attr("y2",function ()  {return front.acutance.yScale(1)})
	
		svga.append("g").attr("id","horizontalGrid")
		d3.select("#horizontalGrid").selectAll(".gridlines")
		    .data(front.acutance.yScale.ticks(5)).enter()
		    .append("line")
		    .attr("class","gridlines")
		    .attr("x1",function ()  {return front.acutance.xScale(front.zmin)})
		    .attr("x2",function ()  {return front.acutance.xScale(front.zmax)})
		    .attr("y1",function (d) {return front.acutance.yScale(d)})
		    .attr("y2",function (d) {return front.acutance.yScale(d)})
		};
		
	    var drawAxes = function () {
		// draw axes
	
		svga = d3.select("#acutanceGroup")
	
		// define xAxis and yAxis
		var xAxis  = d3.svg.axis().scale(front.acutance.xScale).orient("bottom");
		var yAxis  = d3.svg.axis().scale(front.acutance.yScale).orient("left").ticks(5);
	
		svga.append("g")
		    .attr("class", "x plotaxis")
		    .attr("transform", "translate(0," + front.acutance.height + ")")
		    .call(xAxis);
	      
		svga.append("g")
		    .attr("class", "y plotaxis")
		    //.attr("transform","translate("+front.acutance.width/2+",0)")
		    .attr("transform","translate(0,0)")
		    .call(yAxis)
		    .append("text")
		    .attr("transform", "rotate(-90)")
		    .attr("y", 6)
		    .attr("dy", ".71em")
		    .style("text-anchor", "end");
	    };
	    
	    var drawPlot = function () {
		
		// the way the data is structured poses a problem for the d3 enter/exit
		// methodology (ie, doesnt work!). instead it seems easier to simply
		// remove all the children groups of #g2_group and replot all front.regions
		svga = d3.select('#acutanceGroup');
	    
		// clear the old plot
		svga.select("#acutancePlot").remove()
		svga.append("g").attr("id","acutancePlot")

		// make the new plot
		d3.select("#acutancePlot").selectAll("path")
		    .data([front.acutance.data])
		    .enter()
		    .append("path")
		    .attr("d",function (d) {return front.acutance.Line(d)})
		    .attr("fill","none")
		    .attr("stroke","black")
		    .attr("stroke-width",2);
		    
		// draw the marker ball
		d3.select("#acutancePlot").append("circle")
		    .attr("cx",function () {return front.acutance.xScale(front.zmin)})
		    .attr("cy",function () {return front.acutance.yScale(front.acutance.data[0].y)})
		    .attr("r",5)
		    .attr("fill","red")
		    .attr("id","acutanceMarker");
	    };
	    
	    var frontAcutance = function () {

		var afterLoad = function (error,data) {
		    
		    // parse the data
		    front.acutance.data = data.map(function (d) {return {x:parseFloat(d.z),y:parseFloat(d.acutance)}})
		    
		    // define new scales
		    front.acutance.xScale = d3.scale.linear().range([0, front.acutance.width]).domain([front.zmin,front.zmax]);
		    front.acutance.yScale = d3.scale.linear().range([front.acutance.height, 0]).domain([0,1]);
		    
		    // now draw the plot; component functions are broken up for readability
		    resetSVG();
		    drawGrids();
		    drawAxes();
		    drawPlot();
		    
		    // bring up the opacity of the group
		    d3.select("#acutance").transition().duration(250).style("opacity",1);
		    
		};

		queue()
		    .defer(d3.csv, '/static/imaging/csv/acutance_session'+front.sessionId+'_id'+front.propagationId+'.csv')
		    .await(afterLoad);
	    };
	    
	    var backend = function (callback) {
		
		// send the region coordinates to the backend along with
		// the zoom value; from this, the backend will calculate
		// the pixel coordinates of the region and do the back
		// propagation
		
		// draw the clock
		d3.select("#svgaperture").selectAll(".clockface").transition().duration(250).style("opacity",1)
		
		// lock user actions
		ts = new Date().getTime()
		lock(ts);
		
		// get the state of the apodize box. ugly hack!
		var ap
		if ($('#apodize').is(":checked"))  {ap = 1}
		else {ap = 0}
		
		// format the request
		var url  = "fth/propagate"
		var info = {'rmin':1.*front.region.coords.rmin/front.svgHeight,
			    'rmax':1.*front.region.coords.rmax/front.svgHeight,
			    'cmin':1.*front.region.coords.cmin/front.svgWidth,
			    'cmax':1.*front.region.coords.cmax/front.svgWidth,
			    'zoom':front.zoom,'energy':e,'zmin':z1,'zmax':z2,'pitch':p,'apodize':ap};
		
		$.getJSON(url, info,
		    function(json_data) {
			console.log(json_data.result)
			front.propagationId = json_data.propagationId
			
			// load the propagation strip. set the background correctly.
			front.bp_strip = new Image()
			front.bp_strip.onload = function () {
			    var w = this.width, h = this.height;
			    front.g = w/300;
			    d3.select("#apertureimage")
				.attr("width",w)
				.attr("height",h)
				.attr("x",0)
				.attr("xlink:href",front.bp_strip.src);
			    d3.select("#svgaperture").selectAll(".clockface").transition().duration(150).style("opacity",0);
			    unlock(ts);
			    callback(null)
			};
			
			var path = 'static/imaging/images/bp_session'+front.sessionId+'_id'+front.propagationId+'.jpg'
			front.bp_strip.src = path;
		    }
		);
	    };

	    var frontend = function (error) {
		
		// log any error
		if (error != null){
		    console.log("error returning from propagate")
		    console.log(error)}
		
		frontSlider();
		frontAcutance();
	    }

	    var e  = parseFloat($('#energy').val());
	    var p  = parseFloat($('#pitch').val());
	    var z1 = parseFloat($('#zmin').val());
	    var z2 = parseFloat($('#zmax').val());
	    
	    var typesOK = !(isNaN(e) || isNaN(p) || isNaN(z1) || isNaN(z2))
    
	    // save zmin and zmax to the front tracker. if zmin > zmax, reverse the assignment
	    var z3 = parseInt(z1);
	    var z4 = parseInt(z2);
	    if (z3 > z4) {front.zmin = z4; front.zmax = z3};
	    if (z4 > z3) {front.zmin = z3; front.zmax = z4};
	    if (z4 === z3) {typesOK = false};

	    if (e != '' && p != '' && z1 != '' && z2 != '' && front.hasdata && typesOK) {queue().defer(backend).await(frontend)};
	};
    },
    
    brushed: function () {
	var value = front.slider.brush.extent()[0];
	if (d3.event.sourceEvent) { // not a programmatic event
	    value = front.slider.sx.invert(d3.mouse(this)[0]);
	    front.slider.brush.extent([value, value]);
	}

	var z0  = Math.floor(value);
	var idx = z0-front.zmin;
	var ay  = front.acutance.data[idx].y
	
	// move the slider
	d3.select("#handle").attr("cx",front.slider.sx(value));

	var ix = idx%front.g, iy = Math.floor(idx/front.g)
	$("#apertureimage").attr('x',-300*ix).attr('y',-300*iy);
	
	// move the acutance ball and lines
	d3.select("#acutancemarker")
	    .attr("cx",function () {return front.acutance.xScale(z0)})
	    .attr("cy",function () {return front.acutance.yScale(ay)})
	    
	var connect = [[{x:front.zmin,y:ay},
			{x:z0,         y:ay},
			{x:z0,         y:0}],];

	var x = d3.select("#acutancePlot").selectAll("#connect").data(connect)
	
	// when the line is new, set its attributes
	x.enter().append("path")
	    .attr("fill","none")
	    .attr("stroke","red")
	    .attr("stroke-width",1)
	    .attr("id","connect")
	    .attr("stroke-dasharray","5,5")
	    
	// new or old, set the vertices
	x.attr("d",function (d) {return front.acutance.Line(d)});
    },	
};

var start = function () {

    var startHologram = function () {
    
	// add svgs to hologram panel
	d3.select("#hologram").append("svg")
	    .attr("id","svghologram")
	    .attr("width",300)
	    .attr("height",300)
	    
	d3.select("#svghologram").append("image")
	    .attr("id","hologramimage")
	    .attr("width", function() {return front.zooms*300;})
	    .attr("height","100%")
	    .attr("x",0)
	    .attr("y",0)
	    .attr('xlink:href',front.zoom_strip.src)
	    //.attr('transform','scale(2)') // zoom!
	    //.attr('transform','translate(-150)') // move around
	    //.attr('transform','scale(2),translate(-20)') // do both
	    
	d3.select("#svghologram").append("g")
	    .attr("id","hologrambuttons")
	    
	var hrects = [
	    {x:10,y:10,action:"zoomIn"},
	    {x:35,y:10,action:"zoomOut"},
	    {x:270,y:10,action:"propagate"}];
	
	var paths = [
	    {points:[{x:14,y:20},{x:26,y:20}],action:"zoomIn"},
	    {points:[{x:20,y:14},{x:20,y:26}],action:"zoomIn"},
	    {points:[{x:39,y:20},{x:51,y:20}],action:"zoomOut"},
	    {points:[{x:274,y:20},{x:286,y:20}],action:"propagate"},
	    {points:[{x:282,y:16},{x:286,y:20},{x:282,y:24}],action:"propagate"}];
	
	var lineFunc = d3.svg.line()
		.interpolate("linear")
		.x(function(d) { return d.x; })
		.y(function(d) { return d.y; });
	
	d3.select("#hologrambuttons").selectAll("rect")
	    .data(hrects)
	    .enter()
	    .append("rect")
	    .attr("x",function(d)  {return d.x})
	    .attr("y",function(d)  {return d.y})
	    .attr("action",function(d) {return d.action})
	    .attr("height",20)
	    .attr("width",20)
	    .attr("rx",3)
	    .attr("ry",3)
	    .style("fill","white")
	    .attr("class","hologrambuttons")
	    .on("click",function () {if (Object.keys(backendTasks).length === 0) {userFunctions.holoclick(d3.select(this).attr("action"))}});
	
	d3.select("#hologrambuttons").selectAll("path")
	    .data(paths).enter().append("path")
	    .attr("d",function(d) {return lineFunc(d.points)})
	    .attr("action",function(d) {return d.action})
	    .attr("class","hologrambuttons")
	    .style("fill","none")
	    .on("click",function () {if (Object.keys(backendTasks).length === 0) {userFunctions.holoclick(d3.select(this).attr("action"))}});
	};
	
    var startAperture = function () {
	// add svgs to aperture panel
	d3.select("#aperture").append("svg")
	    .attr("id","svgaperture")
	    .attr("width",300)
	    .attr("height",300)
	    
	// add the width attribute when the image is loaded
	d3.select("#svgaperture").append("image")
	    .attr("id","apertureimage")
	    .attr("height",300);
    }
    
    var drawClock = function (where) {

	// size and center
	var rInner = 42, rOuter = 50;

	var clockColor 
	if (front.havegpu) {clockColor = "green"}
	else {clockColor = "#FF7F50"}
	console.log(clockColor)
	
	var circles = [
	    {cx:0,cy:0,r:rOuter,f:clockColor,s:"black",sw:0},
	    {cx:0,cy:0,r:rInner,f:"white",s:"black",sw:0},
	    {cx:0,cy:0,r:5,f:clockColor,s:"black",sw:0}]
	
	for (var n=0;n<12;n++) {
	    var nc = {f:clockColor,s:"black",sw:0,r:2}
	    nc.cx = rInner*0.8*Math.cos(Math.PI*2/12*n);
	    nc.cy = rInner*0.8*Math.sin(Math.PI*2/12*n);
	    circles.push(nc)}
	    
	d3.select('#'+where).append("g")
	    .attr("id",where+"Clock")
	    .attr("transform","translate(150,150)");
	    
	cf = d3.select("#"+where+"Clock");
	
	cf.selectAll("circle").data(circles).enter()
	    .append("circle")
	    .attr("class","clockface")
	    .attr("cx",function(d){return d.cx})
	    .attr("cy",function(d){return d.cy})
	    .attr("r",function(d) {return d.r})
	    .style("fill",function(d) {return d.f})
	    .style("stroke",function(d){return d.s})
	    .style("stroke-width",function(d){return d.sw});
	    
	var hands = [[{x:0,y:0},{x:0,y:30}],[{x:0,y:0},{x:30,y:0}]];

	var handLine = d3.svg.line()
	    .interpolate("linear")
	    .x(function(d) { return d.x; })
	    .y(function(d) { return d.y; });

	cf.selectAll("path").data(hands).enter().append("path")
	    .attr("class","clockface")
	    .attr("d", function(d) {return handLine(d)})
	    .style("stroke",clockColor)
	    .style("stroke-width",2);
    };
    
    var startSlider = function (z) {

	// add object to front
	front.slider = {}
	front.slider.margins = {top: 20, right: 30, bottom: 30, left: 50};
	front.slider.width   = 604 - front.slider.margins.left - front.slider.margins.right;
	front.slider.height  = 50 - front.slider.margins.bottom - front.slider.margins.top;
	
	var sg = d3.select("#slider").append("svg")
	    .attr("width", front.slider.width + front.slider.margins.left + front.slider.margins.right)
	    .attr("height", front.slider.height + front.slider.margins.top + front.slider.margins.bottom)
	    .attr("id","svgslider")
    };
    
    var startAcutance = function () {

	front.acutance = {}
	front.acutance.margins = {top: 20, right: 30, bottom: 30, left: 50};
	front.acutance.width   = 604 - front.acutance.margins.left - front.acutance.margins.right;
	front.acutance.height  = 230 - front.acutance.margins.bottom - front.acutance.margins.top;
	
	var sg = d3.select("#acutance").append("svg")
	    .attr("width", front.acutance.width + front.acutance.margins.left + front.acutance.margins.right)
	    .attr("height", front.acutance.height + front.acutance.margins.top + front.acutance.margins.bottom)
	    .attr("id","svgacutance")
	    
	front.acutance.Line = d3.svg.line()
	    .interpolate("linear")
	    .x(function(d) { return front.acutance.xScale(d.x); })
	    .y(function(d) { return front.acutance.yScale(d.y); });   
    };
    
    var drawRegion = function () {
	// draw a draggable region. coordinates are relative to the svg box, and
	// must be transformed by the backend to bring them into accordance with
	// the current zoom level.
	front.region = {}
	var rs = front.regionSize, ds = front.dragSize, ss = front.selectSize, tlds=front.tlds;
	front.region.coords = {'rmin':tlds/2-rs/2,'rmax':tlds/2+rs/2,'cmin':tlds/2-rs/2,'cmax':tlds/2+rs/2};
	var tc = front.region.coords

	// define the relevant common attributes of the boxes
	d3.select("#svghologram").append("g").attr("id","hologramRegion")
	var allBoxes = [
	    {h:rs, w: rs, x:tc.cmin,    y:tc.rmin,    c:"mainRegion", curs:"move"},
	    {h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin+rs, c:"lowerRight", curs:"se-resize"},
	    {h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin-ds, c:"upperRight", curs:"ne-resize"},
	    {h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin+rs, c:"lowerLeft",  curs:"sw-resize"},
	    {h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin-ds, c:"upperLeft",  curs:"nw-resize"}];

	var group = d3.select("#hologramRegion")
    
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
		.style("fill","white")
		.style("fill-opacity",0)
		.style("cursor",thisBox.curs)
		
	    //newBox.call(drag)

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
			dragFunctions.updateBoxes(t.attr("location"))}
		    })
		.on("dragend",function () {
		    // when dragging has finished, update the region information both here
		    // and in the python backend. then recalculate and replot.
		    if (Object.keys(backendTasks).length === 0) {
			dragFunctions.updateRegion()
			}
		});
	    
	    // attach the dragging behavior
	    newBox.call(drag);
	}
    };
    
    var backend = function (callback) {
	
	// query the backend and get the dataid and the number of zoom levels
	$.getJSON("fth/query", {},
	    function (data) {
		
		console.log(data);
		
		// pull data out of the returned json
		front.sessionId = data.sessionId
		front.dataId = data.dataId;
		front.zoom = 0;
		front.zooms = data.zooms;
		front.hasdata = true;
		front.havegpu = data.hasgpu;

		// load the zoom strip. set the background correctly.
		front.zoom_strip = new Image()
		front.zoom_strip.onload = function () {callback(null)}
		var path = 'static/imaging/images/zooms_session'+front.sessionId+'_id'+front.dataId+'_'+'0.8_logd.jpg';
		front.zoom_strip.src = path;
	    }
	)};
	
    var frontend = function (error) {

	startHologram();
	startAperture();
	startAcutance();
	startSlider();
	drawClock("svghologram");
	drawClock("svgaperture");
	drawRegion();
	
	// remove the clock
	d3.select("#svghologram").selectAll(".clockface").style("opacity",0)
	    
	// move the region selector to front
	t = d3.select("#hologramRegion")[0][0];
	console.log(t)
	t.parentNode.appendChild(t);
	    
	// make the hologram buttons visible
	d3.selectAll(".hologrambuttons").style("opacity",1);
    
    };
    
    queue().defer(backend).await(frontend);
};

start()