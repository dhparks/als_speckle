var backendTasks = {}

// front-end variables are stored in the "front" object
var front = {};

// configurable element sizes
front.sizes = {};
front.sizes.dragger = 5;
front.sizes.region  = 30;
front.sizes.window  = 300;
front.sizes.tlds    = 300;
front.sizes.scaler  = front.sizes.window/300; // don't change this

// variables pertaining to the hologram element
front.hologram = {}
front.hologram.hasdata = false;
front.hologram.translateX = 0;
front.hologram.translateY = 0;
front.hologram.staticTranslateX = 0;
front.hologram.staticTranslateY = 0;
front.hologram.zoomFactor = 1.25
front.hologram.unlocked = true;

// variables for the slider element
front.slider = {}

// variables which describe the data. populated later.
front.data = {};
front.data.exists = false;

//var front = {dragSize:5,regionSize:30,svgHeight:300,svgWidth:300,tlds:300,filename:null,hasdata:false,zoom:1,translateX:0,translateY:0,staticTranslateX:0,staticTranslateY:0}

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
	var ds = front.sizes.dragger, wh = front.sizes.window, ww = front.sizes.window;
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
	    x1b = Math.min(x2-ds,Math.max(0,    dx));
	    x2b = Math.max(x1+ds,Math.min(ww-ds,dx));
	    y1b = Math.min(y2-ds,Math.max(0,    dy));
	    y2b = Math.max(y1+ds,Math.min(wh-ds,dy));
	    
	    // calculate the new values
	    if (what === 'upperLeft')  {new_x1 = x1b; new_x2 = x2;  new_y1 = y1b; new_y2 = y2;}
	    if (what === 'upperRight') {new_x1 = x1;  new_x2 = x2b; new_y1 = y1b; new_y2 = y2;}
	    if (what === 'lowerLeft')  {new_x1 = x1b; new_x2 = x2;  new_y1 = y1;  new_y2 = y2b;}
	    if (what === 'lowerRight') {new_x1 = x1;  new_x2 = x2b; new_y1 = y1;  new_y2 = y2b;}
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
	front.hologram.region.coords.rmin = parseInt(group.select('[location="mainRegion"]').attr("y"));
	front.hologram.region.coords.cmin = parseInt(group.select('[location="mainRegion"]').attr("x"));
	front.hologram.region.coords.rmax = parseInt(group.select('[location="lowerRight"]').attr("y"));
	front.hologram.region.coords.cmax = parseInt(group.select('[location="lowerRight"]').attr("x"));   

	// region information only goes to python on click of propagate button
	},
	
    updateBackground: function () {

	// update the front.translateX and front.translateY parameters
	// with the information coming from the drag event; then
	// update the image transform properties
	
	var dx = d3.event.x, dy = d3.event.y;
	front.hologram.translateX = front.hologram.staticTranslateX+dx/front.hologram.zoom;
	front.hologram.translateY = front.hologram.staticTranslateY+dy/front.hologram.zoom;
	var str = "scale("+front.hologram.zoom+") translate("+front.hologram.translateX+","+front.hologram.translateY+")"
	d3.select('#hologramImage').attr("transform",str)
	
    },
    
    // this is the whole d3 drag behavior for the region box
    boxDragBehavior: d3.behavior.drag()
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
			    }),
			
    // this is the drag behavior for the background image
    imgDragBehavior: d3.behavior.drag()
			.origin(function() { 
			    var t  = d3.select(this);
			    console.log("imgdrag")
			    return {x: t.attr("x"),y: t.attr("y")};})
			.on("drag", function() {
			    if (Object.keys(backendTasks).length === 0 && front.hologram.unlocked) {
			    var t = d3.select(this);
			    dragFunctions.updateBackground()}
			    })
			.on("dragend",function () {
			    // when dragging has finished, update the javascript information about the position
			    if (Object.keys(backendTasks).length === 0 && front.hologram.unlocked) {
				front.hologram.staticTranslateX = front.hologram.translateX;
				front.hologram.staticTranslateY = front.hologram.translateY;
				}
			    }),
}

var userFunctions = {

    validateForm: function (id) {
	
	var who  = document.getElementById(id);
	var what = who.value;
	if (isNaN(what))  {who.className="fieldr"};
	if (!isNaN(what)) {who.className="fieldg"};
	if (what==='')    {who.className="field0"};
    },
    
    convertCoord: function (val) {
	console.log(val)
	return {'x':(val-front.hologram.staticTranslateX*front.hologram.zoom)/front.hologram.zoom,
		'y':(val-front.hologram.staticTranslateY*front.hologram.zoom)/front.hologram.zoom}
    },

    holoclick: function (what) {
	
	// deal with clicks to buttons on the hologram panel. if zoomIn or
	// zoomOut, increment/decrement the zoom level and load a new
	// background image. if propagate, call the propagate routine in the backend.
	
	var zooms = ['zoomIn','zoomOut']
	if (zooms.indexOf(what) > -1) {
	    
	    console.log(what)

	    // the current center coordinates  given by:
	    var cc = userFunctions.convertCoord(front.sizes.window/2);
	    console.log(cc)
	    
	    // update the zoom factor
	    if (what === 'zoomIn')  {front.hologram.zoom *= front.hologram.zoomFactor}
	    if (what === 'zoomOut') {front.hologram.zoom /= front.hologram.zoomFactor}
	    
	    // calculate the new translation values to keep the current center
	    front.hologram.staticTranslateX = front.sizes.window/(2*front.hologram.zoom)-cc.x;
	    front.hologram.staticTranslateY = front.sizes.window/(2*front.hologram.zoom)-cc.y;
	    
	    // apply the updated transformation
	    var str = "scale("+front.hologram.zoom+") translate("+front.hologram.staticTranslateX+","+front.hologram.staticTranslateY+")"
	    d3.select('#hologramImage').attr("transform",str);
	};

	if (what === 'lockBg') {
	    
	    // 3 changes: cursor; background of button; logical state front.hologram.unlocked
	    
	    if  (front.hologram.unlocked) {var curs = "default"; fill = "grey"};
	    if (!front.hologram.unlocked) {var curs = "crosshair"; fill = "white"};
	    
	    $("#hologramimage").css("cursor",curs);
	    $("#lockBg").css("fill",fill)
	    front.hologram.unlocked = !front.hologram.unlocked;

	};
	
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
		    front.acutance.yScale = d3.scale.linear().range([front.acutance.height,0]).domain([0,1]);
		    
		    // now draw the plot; component functions are broken up for readability
		    resetSVG();
		    drawGrids();
		    drawAxes();
		    drawPlot();
		    
		    // bring up the opacity of the group
		    d3.select("#acutance").transition().duration(250).style("opacity",1);
		    
		    console.log('loaded acutance')
		    
		};

		queue()
		    .defer(d3.csv, '/static/imaging/csv/acutance_session'+front.data.sessionId+'_id'+front.data.propagationId+'.csv')
		    .await(afterLoad);
	    };
	    
	    var backend = function (callback) {
		
		// send the region coordinates to the backend along with
		// the zoom value; from this, the backend will calculate
		// the pixel coordinates of the region and do the back
		// propagation

		// change the user cursor tp busy indicator
		$("body").css("cursor", "progress");
		
		// lock user actions
		ts = new Date().getTime()
		lock(ts);
		
		// get the state of the apodize box. ugly hack!
		var ap
		if ($('#apodize').is(":checked"))  {ap = 1}
		else {ap = 0}
		
		// format the request
		var url  = "fth/propagate"
		var info = {'rmin':userFunctions.convertCoord(front.hologram.region.coords.rmin).y,
			    'rmax':userFunctions.convertCoord(front.hologram.region.coords.rmax).y,
			    'cmin':userFunctions.convertCoord(front.hologram.region.coords.cmin).x,
			    'cmax':userFunctions.convertCoord(front.hologram.region.coords.cmax).x,
			    'energy':e,'zmin':z1,'zmax':z2,'pitch':p,'apodize':ap,'window':front.sizes.window};
		
		$.getJSON(url, info,
		    function(json_data) {
			
			console.log(json_data.result)
			front.data.propagationId = json_data.propagationId;
			front.aperture.frameSize = json_data.frameSize;
			front.aperture.scale     = front.sizes.window/front.aperture.frameSize;
			
			// load the propagation strip. set the background correctly.
			front.aperture.bp_strip = new Image()
			
			front.aperture.bp_strip.onload = function () {
			    var w = this.width, h = this.height;
			    front.aperture.gridSize = w/front.aperture.frameSize
			    d3.select("#apertureimage")
				.attr("width",this.width)
				.attr("height",this.height)
				.attr("xlink:href",front.aperture.bp_strip.src)
				.attr("transform","scale("+front.aperture.scale+")");
			    unlock(ts);
			    callback(null)
			};
			$("body").css("cursor", "default");
			var path = 'static/imaging/images/bp_session'+front.data.sessionId+'_id'+front.data.propagationId+'.jpg'
			front.aperture.bp_strip.src = path;
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
	    
	    console.log(e,p,z1,z2)
	    
	    var typesOK = !(isNaN(e) || isNaN(p) || isNaN(z1) || isNaN(z2))
	    
	    console.log(typesOK)
    
	    // save zmin and zmax to the front tracker. if zmin > zmax, reverse the assignment
	    var z3 = parseInt(z1);
	    var z4 = parseInt(z2);
	    if (z3 > z4) {front.zmin = z4; front.zmax = z3};
	    if (z4 > z3) {front.zmin = z3; front.zmax = z4};
	    if (z4 === z3) {typesOK = false};

	    if (e != '' && p != '' && (z1 != '' || z1 === 0) && (z2 != '' || z2 === 0) && front.data.exists && typesOK) {
		queue().defer(backend).await(frontend)};
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

	// raster the image
	var ix  = idx%front.aperture.gridSize, iy = Math.floor(idx/front.aperture.gridSize)
	var str = 'scale('+front.aperture.scale+') translate(-'+front.aperture.frameSize*ix+',-'+front.aperture.frameSize*iy+')'
	$("#apertureimage").attr('transform',str);
	
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
	    .attr("width",front.sizes.window)
	    .attr("height",front.sizes.window)
	    
	// calculate the correct zoom factor for to make
	// the image fit in the image window
	front.hologram.zoom = front.sizes.window/front.hologram.image.width;
	
	// add the hologram image
	d3.select("#svghologram").append("image")
	    .attr("id","hologramimage")
	    .attr("width", front.hologram.image.width)
	    .attr("height",front.hologram.image.height)
	    .attr('xlink:href',front.hologram.image.src)
	    .attr('transform','scale ('+front.hologram.zoom+')')
	    .style("cursor","crosshair")
	    .call(dragFunctions.imgDragBehavior)
	    
	console.log('here')
	    
	d3.select("#svghologram").append("g")
	    .attr("id","hologrambuttons")
	    
	var hrects = [
	    {x:10,y:10,action:"zoomIn"},
	    {x:35,y:10,action:"zoomOut"},
	    {x:60,y:10,action:"lockBg"},
	    {x:270,y:10,action:"propagate"}];
	
	var paths = [
	    {points:[{x:14,y:20},{x:26,y:20}],action:"zoomIn"},
	    {points:[{x:20,y:14},{x:20,y:26}],action:"zoomIn"},
	    
	    {points:[{x:39,y:20},{x:51,y:20}],action:"zoomOut"},
	    
	    {points:[{x:64,y:27},{x:76,y:27},{x:76,y:19},{x:64,y:19},{x:64,y:27}],action:"lockBg"},
	    {points:[{x:68,y:19},{x:68,y:13},{x:72,y:13},{x:72,y:19},{x:68,y:19}],action:"lockBg"},
	    
	    {points:[{x:274,y:20},{x:286,y:20}],action:"propagate"},
	    {points:[{x:282,y:16},{x:286,y:20},{x:282,y:24}],action:"propagate"}];
	
	var lineFunc = d3.svg.line()
		.interpolate("linear")
		.x(function(d) { return d.x; })
		.y(function(d) { return d.y; });
		
	console.log('here')
	
	d3.select("#hologrambuttons").selectAll("rect")
	    .data(hrects)
	    .enter()
	    .append("rect")
	    .attr("id",function (d) {return d.action})
	    .attr("x",function(d)  {return d.x})
	    .attr("y",function(d)  {return d.y})
	    .attr("action",function(d) {return d.action})
	    .attr("height",20*front.sizes.scaler)
	    .attr("width",20*front.sizes.scaler)
	    .attr("rx",3*front.sizes.scaler)
	    .attr("ry",3*front.sizes.scaler)
	    .style("fill","white")
	    .attr("class","hologrambuttons")
	    .on("click",function () {if (Object.keys(backendTasks).length === 0) {userFunctions.holoclick(d3.select(this).attr("action"))}});
	
	d3.select("#hologrambuttons").selectAll("path")
	    .data(paths).enter().append("path")
	    .attr("d",function(d) {return lineFunc(d.points)})
	    .attr("action",function(d) {return d.action})
	    .attr("class","hologrambuttons")
	    //.style("fill","none")
	    .on("click",function () {if (Object.keys(backendTasks).length === 0) {userFunctions.holoclick(d3.select(this).attr("action"))}});
	};
	
    var startAperture = function () {
	
	front.aperture = {}
	
	// add svgs to aperture panel
	d3.select("#aperture").append("svg")
	    .attr("id","svgaperture")
	    .attr("width",front.sizes.window)
	    .attr("height",front.sizes.window)
	    
	// add the width attribute when the image is loaded
	d3.select("#svgaperture").append("image")
	    .attr("id","apertureimage")
	    .attr("height",front.sizes.window);
    }
    
    var startSlider = function (z) {

	// add object to front
	
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
	front.acutance.width   = 2*front.sizes.window + 4 - front.acutance.margins.left - front.acutance.margins.right;
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
	// must be transformed by the backend
	front.hologram.region = {}
	var rs = front.sizes.region, ds = front.sizes.dragger, tlds=front.sizes.window;
	front.hologram.region.coords = {'rmin':tlds/2-rs/2,'rmax':tlds/2+rs/2,'cmin':tlds/2-rs/2,'cmax':tlds/2+rs/2};
	var tc = front.hologram.region.coords

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
	    
	    // attach the dragging behavior
	    newBox.call(dragFunctions.boxDragBehavior);
	}
    };
    
    var backend = function (callback) {
	
	// we query the backend. optionally, we can pass a "resize" argument in
	// the json to force the backend to resize the hologram to a
	// particular size. if this argument is not supplied, the image
	// we get back is the one generated by the backend on loading, and is
	// the same size as the data.

	// query the backend and get the dataid and the number of zoom levels
	$.getJSON("fth/query", {},
	    function (data) {
		
		console.log(data);
		
		// pull data out of the returned json
		front.data.exists    = true;
		front.data.dataId    = data.dataId;
		front.data.sessionId = data.sessionId;
		front.data.haveGPU   = data.hasGPU;
		
		// load the image off the server. given its size and the size
		// of the display window, set the zoom factor and translation
		// coordinates correctly
		var path = 'static/imaging/images/ifth_session'+front.data.sessionId+'_id'+front.data.dataId+'_'+'0.8_logd.jpg';
		front.hologram.image        = new Image()
		front.hologram.image.onload = function () {callback(null)}
		front.hologram.image.src    = path;

	    }
	)};
	
    var frontend = function (error) {

	startHologram();
	startAperture();
	startAcutance();
	startSlider();
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