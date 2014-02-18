
var _checkLock = function(tw) {
    var t1 = !gui.isLocked;
    try { var t2 = !gui.components[tw].isLocked; }
    catch (err) {var t2 = true;}
    return (t1 && t2)
}

// *** define the class for draggable backgrounds. currently,  the needs
// of this object type are served by the single parent class and
// there is no need for children
function draggableBackground(where,sizeX,sizeY) {

    this.name  = where
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    
    this.zoom       = 0;
    this.zoomFactor = 1.25
    this.translateX = 0;
    this.translateY = 0;
    this.staticTranslateX = 0;
    this.staticTranslateY = 0;
    this.unlocked = true;
    
    // non-drag methods
    this.draw = function () {
	
	var tw = this.name
	
	var db = d3.behavior.drag()
	    .origin(function () {
		var t = d3.select(this);
		return {x: t.attr("x"), y:t.attr("y")}
		})
	    .on("drag", function () {
		if (_checkLock(tw)) {
		    draggableBackground.prototype.drag(tw) } // why do i have to call x.prototype.f() instead of x.f()?
		})
	    .on("dragend", function () { if (_checkLock(tw)) {
		    draggableBackground.prototype.dragEnd(tw) }
		})
	
	// remove any old svgs if they exist
	d3.select("#"+this.name+'-svg').remove()
	
	// append an svg element to the DIV
	d3.select('#'+this.name)
	    .append("svg")
	    .attr("id",this.name+'-svg')
	    .attr("width",this.sizeX)
	    .attr("height",this.sizeY)
	    
	// append an image to the svg. attach the d3 dragging behavior
	// to the image element only.
	d3.select('#'+this.name+'-svg')
	    .append("image")
	    .attr("id",this.name+'-img')
	    .style("cursor","crosshair")
	d3.select('#'+this.name+'-img').call(db)

    };
    
    this.loadImage = function (path) {
        
	var setAttr = function (t) {
    
	    // calculate the correct scaling factor. assumes square data?
	    if (t.zoom === 0) {t.zoom = gui.sizes.window/t.image.width;}
	
	    // set all the image attributes
	    d3.select('#'+t.name+'-img')
		.attr("width",t.image.width)
		.attr("height",t.image.height)
		.attr("xlink:href",t.image.src)
		.attr('transform',"scale("+t.zoom+") translate("+t.staticTranslateX+","+t.staticTranslateY+")")
	    }

	// load a background off the server and set the image attribute
	var t = this;
	this.image        = new Image()
	this.image.onload = function () {setAttr(t)}
	this.image.src    = path;
    };
    
    this.zoomIn = function () {

	if (this.unlocked) {
    
	    // first get the current center coordinates
	    var cx = (gui.sizes.window/2-this.staticTranslateX*this.zoom)/this.zoom;
	    var cy = (gui.sizes.window/2-this.staticTranslateY*this.zoom)/this.zoom;	
	    
	    // now update the zoom factor
	    this.zoom *= this.zoomFactor
	    
	    // calculate the new translation values to keep the current center
	    this.staticTranslateX = gui.sizes.window/(2*this.zoom)-cx;
	    this.staticTranslateY = gui.sizes.window/(2*this.zoom)-cy;
	    
	    var str = "scale("+this.zoom+") translate("+this.staticTranslateX+","+this.staticTranslateY+")"
	    d3.select('#'+this.name+'-img').attr("transform",str);
	}
    }
    
    this.zoomOut = function () {

	if (this.unlocked) {
	    // first get the current center coordinates
	    var cx = (gui.sizes.window/2-this.staticTranslateX*this.zoom)/this.zoom;
	    var cy = (gui.sizes.window/2-this.staticTranslateY*this.zoom)/this.zoom;	
	    
	    // now update the zoom factor
	    this.zoom /= this.zoomFactor
	    
	    // calculate the new translation values to keep the current center
	    this.staticTranslateX = gui.sizes.window/(2*this.zoom)-cx;
	    this.staticTranslateY = gui.sizes.window/(2*this.zoom)-cy;
	    
	    var str = "scale("+this.zoom+") translate("+this.staticTranslateX+","+this.staticTranslateY+")"
	    d3.select('#'+this.name+'-img').attr("transform",str);
	}
    }
    
    this.lock = function () {
	// 3 changes: cursor; background of button; boolean locked state
	if  (this.unlocked) {var curs = "default"; fill = "grey"};
	if (!this.unlocked) {var curs = "crosshair"; fill = "white"};
	$("#"+this.name+'-img').css("cursor",curs);
	$("#"+this.name+'-lockBg').css("fill",fill)
	this.unlocked = !this.unlocked;
    }
}

draggableBackground.prototype.drag = function (name) {
    var bg  = gui.components[name].background
    if (bg.unlocked) {
	var dx  = d3.event.x;
	var dy  = d3.event.y;
	var z   = bg.zoom;
	var stx = bg.staticTranslateX;
	var sty = bg.staticTranslateY;
	var tx  = stx+dx/z;
	var ty  = sty+dy/z;
    
	bg.translateX = tx;
	bg.translateY = ty;
	var str = "scale("+z+") translate("+tx+","+ty+")"
	
	d3.select('#'+name+'-img').attr("transform",str)
    }
}

draggableBackground.prototype.dragEnd = function (name) {
    
    // update the staticTranslate attributes at the end of dragging
    var bg = gui.components[name].background
    bg.staticTranslateX = bg.translateX;
    bg.staticTranslateY = bg.translateY;
    
}

// *** define the action button prototype.
function actionButton(where,action,coords,icon) {
    
    icons = {
	    'plus':[[{x:4,y:10},{x:16,y:10}],
		    [{x:10,y:4},{x:10,y:16}]],
	    'minus':[[{x:4,y:10},{x:16,y:10}],],
	    'lock': [[{x:4,y:17},{x:16,y:17},{x:16,y:10},{x:4,y:10},{x:4,y:18}],
		    [{x:8,y:10},{x:8,y:3},{x:12,y:3},{x:12,y:7.5}]],
	    'rArrow':[[{x:4,y:10},{x:16,y:10}],
		    [{x:12,y:6},{x:16,y:10},{x:12,y:14}]],
	    'dArrow':[[{x:10,y:4},{x:10,y:16}],
		    [{x:6,y:12},{x:10,y:16},{x:14,y:12}]],
	    'square':[[{x:4,y:4},
		   {x:16,y:4},
		   {x:16,y:16},
		   {x:4,y:16},
		   {x:4,y:4-1}],],
	    'x':[[{x:4,y:4}, {x:16,y:16}],
		   [{x:16,y:4},{x:4, y:16}]],
	    '2Arrow':[[{x:4,y:6},{x:16,y:6}],[{x:12,y:3},{x:16,y:6},{x:12,y:9}],
		     [ {x:4,y:14},{x:16,y:14}],[{x:8,y:11},{x:4,y:14},{x:8,y:17}]],
	    'tophat':[[{x:3,y:14},{x:7,y:14},{x:7,y:6},{x:13,y:6},{x:13,y:14},{x:17,y:14}],],
	    'derivative':[[{x:3,y:10},{x:7,y:10},{x:7,y:4},{x:7,y:10},{x:13,y:10},{x:13,y:16},{x:13,y:10},{x:17,y:10}],]}
	    
    
    this.lineFunc = d3.svg.line().interpolate("linear")
	.x(function(d) { return (d.x+this.coords.x)*gui.sizes.scaler; })
	.y(function(d) { return (d.y+this.coords.y)*gui.sizes.scaler; });
	
    this.draw = function () {

	// first, look for an svg holder given by the form this.where-buttons. if it doesn't exist, create it.
	if (!$('#'+this.where+'-buttons').length) {
	    d3.select('#'+this.where+'-svg').append("svg").attr("id",this.where+"-buttons")
	}
	
	var t = this
	
	// now add the requested button
	d3.select("#"+this.where+'-buttons')
	    .append("rect")
	    .attr("id",this.where+'-'+this.action)
	    .attr("x",this.coords.x)
	    .attr("y",this.coords.y)
	    .attr("height",20*gui.sizes.scaler)
	    .attr("width",20*gui.sizes.scaler)
	    .attr("rx",3*gui.sizes.scaler)
	    .attr("ry",3*gui.sizes.scaler)
	    .style("fill","white")
	    .attr("class","controlbuttons")
	    .on("click",function () {if (!gui.isLocked) {guiFunctions.actionDispatch({'action':t.action,'where':t.where})}})
	
	for (var k=0;k<this.paths.length;k++) {
	    
	    path = this.paths[k];
	    
	    d3.select("#"+this.where+'-buttons')
		.append("path")
		.attr("d",this.lineFunc(path))
		.attr("class","controlbuttons")
		.style("fill","none")
		.on("click",function () {if (!gui.isLocked) {guiFunctions.actionDispatch({'action':t.action,'where':t.where})}});
	}
    }

    this.where  = where
    this.coords = coords
    this.action = action
    this.paths  = icons[icon]
}

// *** draggable regions. these should be instantiated as children,
// not the parent prototype
function draggableRegion() {

    this.protoDraw = function (where,coords,sizes,color,includeSelector) {

	// get the coordinates
	var tc = coords
	var ds = sizes.ds;
	var ws = sizes.ws;
	var rs = sizes.rs;
	var ss = sizes.ss;
	
        // define the relevant common attributes of the boxes
        d3.select("#"+this.where+"-svg").append("g")
	    .attr("id",this.where+'-region-'+this.regionId)
	    .attr("where",this.where)
	    .attr("regionId",this.regionId)
        
        var allBoxes = [
            {h:rs, w: rs, x:tc.cmin,    y:tc.rmin,    c:"mainRegion", curs:"move"},
            {h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin+rs, c:"lowerRight", curs:"se-resize"},
            {h:ds, w: ds, x:tc.cmin+rs, y:tc.rmin-ds, c:"upperRight", curs:"ne-resize"},
            {h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin+rs, c:"lowerLeft",  curs:"sw-resize"},
            {h:ds, w: ds, x:tc.cmin-ds, y:tc.rmin-ds, c:"upperLeft",  curs:"nw-resize"}];
    
        var group = d3.select("#"+this.where+"-region-"+this.regionId);
    
        // make the rectangular elements using d3
        for (var k=0;k<allBoxes.length;k++){
    
            var thisBox = allBoxes[k];
            var newBox  = group.append("rect")
    
            newBox
                .attr("x",thisBox.x)
                .attr("y",thisBox.y)
                .attr("height",thisBox.h)
                .attr("width",thisBox.w)
		.attr("regionId",this.regionId)
                .attr("location",thisBox.c)
		.attr("where",this.where)
                .style("fill",color)
                .style("fill-opacity",0)
                .style("cursor",thisBox.curs)
    
            if (thisBox.c==="mainRegion") {
                newBox.style("stroke",color)
                .style("stroke-width",2)
                .style("stroke-opacity",1);}
                
            if (thisBox.c !="mainRegion") {
                newBox.classed("dragger",true)
                .style("fill-opacity",1);}
            
            // attach the dragging behavior
            newBox.call(db);
        }

        // add the selector, if appropriate
        if (includeSelector) {
	    
	    var t = this
	    
            // make the circular element
            group.append("circle")
                .attr("cx",tc.cmin+rs/2)
                .attr("cy",tc.rmin-ss)
                .attr("r",ss)
                .style("fill",color)
                .style("fill-opacity",0) // need a fill to toggle on clickSelect
                .style("stroke-width",2)
                .style("stroke",color)
                .classed("selecter",true)
                .classed("region",true)
                .classed('interactive',true)
                .attr("regionId",this.regionId)
		.attr("where",this.where)
                .on("click",function () {
		    t.protoToggleSelect(where);
		    guiFunctions.actionDispatch({action:'selectRegion',region:this.regionId,'where':where});})
        }
    }

    this.convertCoords = function () {
	// convert coords from local pixel coords seen in this.coords
	// to data coordinates meaningful to the analysis backend
	
	gcwb = gui.components[this.where].background;

	var stx = gcwb.staticTranslateX;
	var sty = gcwb.staticTranslateY;
	var   z = gcwb.zoom;
	
	var converted = {
	    'rmin': (this.coords.rmin-sty*z)/z,
	    'rmax': (this.coords.rmax-sty*z)/z,
	    'cmin': (this.coords.cmin-stx*z)/z,
	    'cmax': (this.coords.cmax-stx*z)/z
	}

	return converted
    }
    
    this.protoToggleSelect = function (where) {
	
	this.selected = !this.selected;
	
	// select the region in the svg; switch the opacity
	var c = d3.select("#"+where+"-region-"+this.regionId).select("circle");
        c.transition().duration(150).style("fill-opacity",1-c.style("fill-opacity"))
    }
    
    this.protoToggleFill = function (where) {
	var t = this;
	// select the region in the svg; switch the opacity
	var c = d3.select("#"+where+"-region-"+this.regionId).select("[location='mainRegion']");
	var v = 0.5-c.style("fill-opacity");
        c.style("fill-opacity",v)
    }

    this.makeId = function () {
	var date       = (new Date().getTime()).toString()
	this.regionId  = "r"+date.slice(date.length-7,date.length-2);
	this.objectId  = this.regionId
    }
    
    var db = d3.behavior.drag()
            .origin(function() {
		var t = d3.select(this);
		return {x: t.attr("x"),y: t.attr("y")};
		})
            .on("drag",   function()  {
		if (_checkLock) {
		    var t = d3.select(this);
		    draggableRegion.prototype.drag(t.attr("where"),t.attr("regionId"),t.attr("location"));}
		    })
            .on("dragend",function () {
		if (_checkLock) {
		    var t = d3.select(this)
		    draggableRegion.prototype.dragEnd(t.attr("regionId"), t.attr("where"))}
		})

    this.remove = function () {
	// remove from the dom and from gui.components
	d3.select("#"+this.where+"-region-"+this.regionId).remove();
	delete gui.components[this.where].regions[this.regionId];
    }
    
    this.selected  = false;
    
}

draggableRegion.prototype.drag = function (where, regionId, what) {
    
    // this function updates the coordinates of the 6 svg elements which
    // show a region. most of the mathematical complexity arises from the
    // decision to enforce boundary conditions at the edge of the data, so
    // no aspect of the region becomes undraggable.

    var group = d3.select('#'+where+'-region-'+regionId);
    
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
    var ds = gui.sizes.dragger, ss = gui.sizes.selector, wh = gui.sizes.window, ww = gui.sizes.window;
    var dx = d3.event.x, dy = d3.event.y

    // one behavior for dragging .main...
    if (what === 'mainRegion') {

	mrx = dx;      mry = dy;
	ulx = dx-ds;   uly = dy-ds;
	llx = dx-ds;   lly = dy+ch;
	urx = dx+cw;   ury = dy-ds;
	lrx = dx+cw;   lry = dy+ch;
	csx = dx+cw/2; csy = dy-ss;

	// check bounds
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

draggableRegion.prototype.dragEnd = function (regionId, where) {

    // update the region information in the tracker
    var group = d3.select("#"+where+"-region-"+regionId)
    
    r = gui.components[where].regions[regionId]
    r.coords.rmin = parseInt(group.select('[location="mainRegion"]').attr("y"));
    r.coords.cmin = parseInt(group.select('[location="mainRegion"]').attr("x"));
    r.coords.rmax = parseInt(group.select('[location="lowerRight"]').attr("y"));
    r.coords.cmax = parseInt(group.select('[location="lowerRight"]').attr("x"));   

    // sometimes we want to do extra things when the dragging ends. for example,
    // in xpcs we might recalculate g2
    try { guiFunctions.actionDispatch({'action':'regionDragEnd','where':where,'regionId':regionId})}
    catch (err) {};
    
    
    // run the recalculation if that option is selected
    if (gui.project === 'xpcs') {
	if ($('#autocalc').is(":checked")) { userFunctions.recalculateG2() };
    }
}

draggableRegionWhite.prototype = new draggableRegion();
draggableRegionColor.prototype = new draggableRegion();
function draggableRegionWhite(where,sizes,selectable) {
    this.where  = where
    this.color  = "white"
    this.makeId()
    this.coords = {
	'rmin':sizes.window/2-sizes.region/2,
	'rmax':sizes.window/2+sizes.region/2,
	'cmin':sizes.window/2-sizes.region/2,
	'cmax':sizes.window/2+sizes.region/2};
    this.ds = sizes.dragger
    this.ws = sizes.window
    this.rs = sizes.region
    this.ss = sizes.selector
    this.draw = function () {this.protoDraw(this.where,this.coords,{ds:this.ds,ws:this.ws,rs:this.rs,ss:this.ss},this.color,selectable)}
    this.toggleSelect = function () {this.protoToggleSelect(this.where)}
}

function draggableRegionColor(where,sizes) {
    
    // like the FTH dragger, but with color and a selector
    
    this.newColor = function (w) {
	
	// look into the region list. find all the colors.
	// make one that is far away from the others.
	
	var _newHue = function (w) {
	
	    var hues = [];
	    for (region in gui.components[w].regions) {
		try { hues.push(gui.components[w].regions[region].hue);}
		catch (err) {console.log(err)}
	    }
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
	
	// make the new hue; turn it into a color;
	// return both to assign to namespace
	this.hue   = _newHue(w);
	this.color = d3.hsl(this.hue*360,1,0.5).toString();
    }

    this.where  = where
    this.newColor(this.where)
    this.makeId()
    this.coords = {
	'rmin':sizes.window/2-sizes.region/2,
	'rmax':sizes.window/2+sizes.region/2,
	'cmin':sizes.window/2-sizes.region/2,
	'cmax':sizes.window/2+sizes.region/2};
    this.ds = sizes.dragger;
    this.ws = sizes.window;
    this.rs = sizes.region;
    this.ss = sizes.selector;
    this.selected = false;
    this.draw = function () {this.protoDraw(this.where,this.coords,{ds:this.ds,ws:this.ws,rs:this.rs,ss:this.ss},this.color,true)}
    this.toggleSelect = function () {this.protoToggleSelect(this.where)}
    this.toggleFill = function () {this.protoToggleFill(this.where)}
}

// object for rastering backgrounds such as that seen in back propagation
// no current need for subclassing
function rasterBackground(where,sizeX,sizeY) {
    
    this.name  = where
    this.sizeX = sizeX
    this.sizeY = sizeY
    this.rasterValue = 0;
    
    // size of each frame in pixels; needs to be
    // filled in by gui when the data is downloaded.
    this.frameSize = 0;
    
    this.draw = function () {
	// append an svg element to the DIV
	d3.select('#'+this.name)
	    .append("svg")
	    .attr("id",this.name+'-svg')
	    .attr("width",this.sizeX)
	    .attr("height",this.sizeY)
	    
	// append an image to the svg. attach the d3 dragging behavior
	// to the image element only.
	d3.select('#'+this.name+'-svg')
	    .append("image")
	    .attr("id",this.name+'-img')
    }
    
    this.loadImage = function (path, rasterValue) {

	var setAttr = function (t) {
    
	    // calculate the correct scaling factor. assumes square data?
	    t.scale = gui.sizes.window/t.frameSize;
	
	    var w = t.image.width, h = t.image.height;
	    t.gridSize = w/t.frameSize;

	    // set all the image attributes
	    d3.select('#'+t.name+'-img')
		.attr("width",w)
		.attr("height",h)
		.attr("xlink:href",t.image.src)
		.attr('transform','scale ('+t.scale+')')
	    }

	// load a background off the server and set it to the
	// correct image attribute
	var t = this;
	if (typeof(rasterValue) === 'undefined') {rasterValue=this.rasterValue}
	this.image        = new Image()
	this.image.onload = function () {setAttr(t); t.raster(rasterValue)}
	this.image.src    = path
    }
    
    this.raster = function (n) {
	// n is the frame number found in some other action, probably
	// a scrubber or sliderGraph
	
	console.log("raster "+n)
	
	var ix  = n%this.gridSize, iy = Math.floor(n/this.gridSize)
	var str = 'scale('+this.scale+') translate(-'+this.frameSize*ix+',-'+this.frameSize*iy+')'
	$("#"+this.name+'-img').attr('transform',str);
	
	this.rasterValue = n;
	
    }

}

// graph objects. DEFINITELY requires subclassing.
function graph() {

    this.defaultColor = "black";
    this.defaultStrokeWidth  = 2;

    this.init = function (where, sizeX, sizeY) {

	// populate the div with the correct svg
	var gcw  = gui.components[where];
	var gcwm = gcw.margins;
	
	d3.select("#"+where).append("svg")
	    .attr("width",  gcw.width  + gcwm.left + gcwm.right)
	    .attr("height", gcw.height + gcwm.top  + gcwm.bottom)
	    .attr("id",where+"-svg");
    }
    
    this.resetSVG = function (where) {
	
	var w = gui.components[where];

	// first, delete the old plot
	d3.select("#"+where+"-group").remove();
	
	// now, readd the svg group
	d3.select("#"+where+"-svg")
	    .append("g")
	    .attr("transform", "translate(" + w.margins.left + "," + w.margins.top + ")")
	    .attr("id",where+"-group")
    }
    
    this.drawGrids = function (where) {
	
	var w = gui.components[where]
	var svga = d3.select("#"+where+"-group")

	//draw grid lines
	svga.append("g").attr("id",where+"-verticalGrid")
	d3.select("#"+where+"-verticalGrid").selectAll(".gridlines")
	    .data(w.xScale.ticks())
	    .enter()
	    .append("line")
	    .attr("class","gridlines")
	    .attr("x1",function (d) {return w.xScale(d)})
	    .attr("x2",function (d) {return w.xScale(d)})
	    .attr("y1",function ()  {return w.yScale(w.rangeMin)})
	    .attr("y2",function ()  {return w.yScale(w.rangeMax)})

	svga.append("g").attr("id",where+"-horizontalGrid")
	d3.select("#"+where+"-horizontalGrid").selectAll(".gridlines")
	    .data(w.yScale.ticks(5)).enter()
	    .append("line")
	    .attr("class","gridlines")
	    .attr("x1",function ()  {return w.xScale(w.domainMin)})
	    .attr("x2",function ()  {return w.xScale(w.domainMax)})
	    .attr("y1",function (d) {return w.yScale(d)})
	    .attr("y2",function (d) {return w.yScale(d)})
    }
    
    this.drawAxes = function (where,extras) {
	
	// the extras argument is how we distinguish the axes behavior for different plot
	// types. optionally, this could be overridden by the child but this is
	// difficult within the prototype inheritance scheme
	
	var w = gui.components[where]
	var svga = d3.select("#"+where+"-group")
	
	if (typeof(extras) === 'undefined') {extras = {}}
	
	// nticksX
	if ('nticksX' in extras) {
	    var xAxis = d3.svg.axis().scale(w.xScale).orient("bottom").ticks(extras.nticksX);}
	else {
	    console.log("here")
	    var xAxis = d3.svg.axis().scale(w.xScale).orient("bottom").ticks(5);}
	
	// nticksY
	if ('nticksY' in extras) {
	    var yAxis = d3.svg.axis().scale(w.yScale).orient("left").ticks(extras.nticksY);}
	else {
	    var yAxis = d3.svg.axis().scale(w.yScale).orient("left").ticks(5);}
	    
	// yText
	if ('yText' in extras) {var yText = extras.yText;}
	else {var yText = '';}
	
	// xText
	if ('xText' in extras) {var xText = extras.xText;}
	else {var xText = '';}

	svga.append("g")
	    .attr("class", "x plotaxis")
	    .attr("transform", "translate(0," + w.height + ")")
	    .call(xAxis)

	svga.append("g")
	    .attr("class", "y plotaxis")
	    .attr("transform","translate(0,0)")
	    .call(yAxis)
	 
	   
	svga.append("text")
	    .attr("y", w.height-15)
	    .attr("x", w.width-4)
	    .attr("dy", ".71em")
	    .style("text-anchor", "end")
	    .text(xText);
	    
	svga.append("text")
	    .attr("transform", "rotate(-90)")
	    .attr("y", 6)
	    .attr("x",-5)
	    .attr("dy", ".71em")
	    .style("text-anchor", "end")
	    .text(yText);
	    
    }

    this.redraw = function (where, sizeX, sizeY, extras) {
	// break these down into smallers parts so that they
	// can be overridden easily by children graphs
	this.resetSVG(where);
	this.drawGrids(where);
	this.drawAxes(where,extras);
    }

    this.replot = function (where, args) {

	var _getColor = function () {
	    try {color = args.color}
	    catch (err) {color = t.defaultColor}
	    return color;
	}
	
	var _getWidth = function () {
	    try {width = args.width}
	    catch (err) {width = t.defaultStrokeWidth}
	    return width;
	}
	
	var t = this
    
	var gcw = gui.components[where]

	svga = d3.select('#'+where+'-group');
    
	// clear the old plot
	svga.selectAll("#"+where+'-plot').remove()
	svga.append("g").attr("id",where+"-plot")

	// make the new plot
	d3.select("#"+where+"-plot").selectAll("path")
	    .data([gcw.data])
	    .enter()
	    .append("path")
	    .attr("d",function (d) {return gcw.lineFunc(d)})
	    .attr("fill","none")
	    .attr("stroke",_getColor())
	    .attr("stroke-width",_getWidth());
    };

}

// basicGraph is used for non-interactive plots such
// as the rftf plot in cdi. basically the prototype, unaltered.
basicGraph.prototype = new graph();
function basicGraph(where,sizeX,sizeY) {
    this.where = where;
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.plotColor = "black";
    this.strokeWidth = 2;
    this.draw  = function (extras) {this.redraw(this.where,this.sizeX,this.sizeY,extras)}
    this.plot  = function () {this.replot(this.where, {'color':this.plotColor,'width':this.strokeWidth});}
    this.init(this.where,this.sizeX,this.sizeY)
}

// sliderGraph is used for acutance plots; the graph includes
// a slider which (typically) interacts with a rasterBackground
sliderGraph.prototype = new graph();
function sliderGraph(where,sizeX,sizeY) {

    this.where  = where;
    this.sizeX  = sizeX;
    this.sizeY  = sizeY;

    this.draw = function (extras) {
	this.redraw(this.where,this.sizeX,this.sizeY,extras)
	};
	
    this.plot = function () {
	this.replot(this.where);
	this.drawBall();
	};
    
    this.drawBall = function () {
	
	// draw the marker ball; attach dragging behavior
	var g = gui.components[this.where];
	
	var db = d3.behavior.drag()
	    .origin(function () {
		var t = d3.select(this);
		return {x: t.attr("x"), y:t.attr("y")}
		})
	    .on("drag", function () {
		if (_checkLock()) {
		    sliderGraph.prototype.dragBall(this.id.replace('-ball','')) }
		})

	// draw the cross hairs first so that they aren't blocking the ball
	var connect = [
	    [{x:g.domainMin,y:g.data[0].y},{x:g.domainMax,y:g.data[0].y}],
	    [{x:g.domainMin,y:0},{x:g.domainMin,y:1}]
	    ];

	var x = d3.select("#"+this.where+"-plot").selectAll("#connect").data(connect);
	
	x.enter().append("path")
	    .attr("fill","none")
	    .attr("stroke","red")
	    .attr("stroke-width",2)
	    .attr("id","connect")
	    .attr("stroke-dasharray","5,5")
	x.attr("d",function (d) {return g.lineFunc(d)})
		
	// now draw the ball with attached dragging behavior
	var x0 = g.xScale(g.domainMin);
	var y0 = g.yScale(g.data[0].y);
	d3.select("#"+this.where+"-plot").append("circle")
	    .attr("cx", x0)
	    .attr("cy", y0)
	    .attr("r",7)
	    .attr("fill","red")
	    .attr("id",this.where+"-ball")
	    .call(db);
    };
    
    this.init(this.where,this.sizeX,this.sizeY);
}

sliderGraph.prototype.dragBall = function (where) {

    // get the old position
    var b    = d3.select("#"+where+'-ball');
    var oldx = parseInt(b.attr("cx"));

    // transform into new location given d3.event.x
    var newx = oldx+d3.event.dx;
    
    var gcw  = gui.components[where]
    var z    = gcw.xScale.invert(newx)
    var idx  = Math.floor(z-gcw.domainMin)
    var newy = gcw.yScale(gcw.data[idx].y)
    
    // draw at new location
    b.attr("cx",newx).attr("cy",newy)
    
    // update the axis connecting lines
    var connect = [[{x:gcw.domainMin,y:gcw.data[idx].y},{x:gcw.domainMax,y:gcw.data[idx].y}],[{x:z,y:0},{x:z,y:1}]];
    var x = d3.select("#"+where+"-plot").selectAll("#connect").data(connect)
    x.attr("d",function (d) {return gcw.lineFunc(d)})
    
    // additional behavior must be specified in the gui script
    guiFunctions.actionDispatch({'action':'scrub','where':where,'n':idx})
},

// clickerGraph is used for the XPCS plots. data series
// can be clicked to display additional information.
clickerGraph.prototype = new graph();
function clickerGraph(where,sizeX,sizeY,hasFitData) {
    
    this.where = where
    this.sizeX = sizeX
    this.sizeY = sizeY
    this.hasFitData = hasFitData
    this.draw  = function (extras) {this.redraw(this.where,this.sizeX,this.sizeY,extras)}

    // these are default values which could be overridden in
    // the gui if the user wanted to
    this.defaultColor = "black"
    this.defaultStrokeWidth = 2
    
    this.replot = function (dataLocation) {

	var _getColor = function (x) {
	    try {color = x.color;}
	    catch (err) {color = this.defaultColor};
	    return color;
	}
    
	var t    = this.where
	var gcw  = gui.components[t];
	var gcwd = gui.components[t][dataLocation];
	
	// the way the data is structured poses a problem for the d3 enter/exit
	// methodology (ie, doesnt work!). instead it seems easier to simply
	// remove all the children groups of #plotGroup and replot all data
	svgg = d3.select('#'+t+'-group');

	// clear all the old plots
	svgg.selectAll(".dataSeries").remove()

	// d3 doesn't play well with being passed an object instead of an
	// array. therefore, recast gui.intensity.regions into an array
	// of objects with only the necessary data

	var plottables = []
	for (var d in gcwd) {plottables.push(gcwd[d])}

	// each plot group is structured as
	// (parent) group
	//    (child)  path: dataValues
	//    (childs) circles dataValues
	//    (child)  path: fitValues

	var newLines = svgg.selectAll(".dataSeries")
	    .data(plottables)
	    .enter()
	    .append("g")
	    .attr("class","dataSeries")
	    .attr("id", function (d) {return t+"-plotGroup-"+d.objectId;})
	    .style("cursor","pointer")
	    .on("click",function () {
		guiFunctions.actionDispatch({'action':'togglePlot','id':d3.select(this).attr("id")})});

	newLines.append("path")
	    .style("fill","none")
	    .style("stroke-width",this.defaultStrokeWidth)
	    .style("stroke",function (d) {return _getColor(d)})
	    .attr("class","dataPath")
	    .attr("id",function(d) {return t+'-data-'+d.objectId})
	    .attr("d", function(d) {return gcw.lineFunc(d.dataValues); });

	// define a scale for the size of the circles so that they decrease in
	// size as time increases
	
	// add data circles.
	for (var k=0;k<plottables.length;k++) {
	    d3.select('#'+t+"-plotGroup-"+plottables[k].objectId).selectAll(".dataCircle")
		.data(plottables[k].dataValues)
		.enter().append("circle")
		.attr("class","dataCircle")
		.attr("cx",function (d) {return gcw.xScale(d.x)})
		.attr("cy",function (d) {return gcw.yScale(d.y)})
		.attr("r", function (d) {return gcw.rScale(d.x)})
		.style("fill", _getColor(plottables[k]))
	}

	// add the fit lines. these are added last to ensure they are above the circles.
	// in the future, it might be better to draw the fit lines after the click,
	// so that they are on top of everything.
	if (this.hasFitData) {
	    newLines.append("path")
		.style("fill","none")
		.style("stroke","black")
		.style("stroke-width",this.defaultStrokeWidth)
		.style("opacity",0)
		.attr("class","fitPath")
		.attr("d", function(d) {return gcw.lineFunc(d.fitValues); })
		;
	}
    }
	
    this.init(this.where,this.sizeX,this.sizeY)
    
}

function clickerReadout (where,args) {
    // set up the fit parameters readout. this object goes along
    // with a clickerGraph, which needs to already be instantiated in where.
    // position is {x:-5,y:5} by default
    
    this.init = function () {

	d3.select("#"+this.where+"-svg").append("g").attr("id",this.where+"-readout");
    
	d3.select("#"+this.where+"-readout")
	    .append("rect")
	    .attr("x",this.dx).attr("y",this.dy)
	    .attr("opacity",0)
	    .attr("id","txtrect")
	    .attr("fill","white")
	    .style("stroke-width",1)
	    .style("stroke","black")
	    
	d3.select("#"+this.where+"-readout")
	    .append("text")
	    .attr("id",this.where+"-readout-text");
    };
    
    this.update = function (lines) {
	
	// remove the old box, then build the new box
	d3.selectAll(".fitText").remove();
	var txt = d3.select("#"+this.where+"-readout-text");
	txt.selectAll("tspan").data(lines).enter()
	    .append("tspan")
	    .text(function (d) {return d})
	    .attr("class","fitText")
	    .attr("x",0)
	    .attr("dy","1.2em")
	    .style("opacity",0);
	    
	var txt = d3.select("#"+this.where+"-readout");
	txt.style("opacity",1)
	var bbox = txt.node().getBBox();
	var gcw  = gui.components[this.where]
	
	if (this.position === 'upper-left') {
	    var tbx = gcw.margins.left+this.padding
	    var tby = gcw.margins.top+this.padding;
	}
	
	if (this.position === 'lower-left') {
	    var tbx = gcw.margins.left+this.padding
	    var tby = gui.sizes.window-bbox.height-gcw.margins.bottom-this.padding;
	}
	
	if (this.position === 'upper-right') {
	    var tbx = gui.sizes.window-bbox.width-gcw.margins.right-this.padding;
	    var tby = gcw.margins.top+this.padding;
	}
	
	if (this.position === 'lower-right') {
	    var tbx = gui.sizes.window-bbox.width-gcw.margins.right-this.padding;
	    var tby = gui.sizes.window-bbox.height-gcw.margins.bottom-this.padding;
	}
	
	d3.select("#"+this.where+"-readout").attr("transform","translate("+tbx+","+tby+")");
	d3.select("#txtrect").attr("width",bbox.width).attr("height",bbox.height);
	d3.selectAll(".fitText").style("opacity",1);
	d3.select("#txtrect").style("opacity",1);
    }

    this.hide = function () {
	d3.select("#"+this.where+"-readout").style("opacity",0)
    };
    
    this.where = where
    
    // defaults and optional args
    this.dx = -5;
    try {this.dx = args.dx}
    catch(err) {};
    
    this.dy = 5;
    try {this.dy = args.dy}
    catch(err) {};
    
    this.position = 'lower-left'
    try {this.position = args.position}
    catch(err) {};
    
    this.padding = 10
    try {this.padding = args.padding}
    catch(err) {};
    
    this.init()
    
    
    };

// *** breadcrumb
var breadcrumb = function (where, id) {

    this.defaultPositioner = function () {
	var n = Object.keys(gui.components[this.where].crumbs).length
	return {cy:15, cx:(1+this.rmax)+(n-1)*(2+this.rmax+this.rmin)}
    }
    
    this.positioner = function () {
	try { pos = gui.components[this.where].breadcrumbPosition() }
	catch (err) {pos = this.defaultPositioner()}
	return pos
    }

    this.draw = function () {
	
	pos = this.positioner();
	
	// add a new breadcrumb to this.where
	d3.select("#"+this.where+"-svg").append("circle")
	    .attr("r",this.rmin)
	    .attr("cy",pos.cy)
	    .attr("cx",pos.cx)
	    .attr("class","breadcrumb")
	    .attr("id",this.svg.replace('#',''))
	    .attr("where",this.where)
	    .attr("rId",this.rId)
	    .on("click",function () {
		var t = d3.select(this)
		guiFunctions.actionDispatch({'action':'breadcrumb','id':t.attr("rId"),'where':t.attr("where")});})
    };
    
    this.enlarge = function (color) {
	if (typeof(color)==="undefined") {color = this.defaultColor}
	d3.select(this.svg).attr("r",this.rmax).style("fill",color)
	};
	
    this.shrink  = function (color) {
	if (typeof(color)==="undefined") {color = this.defaultColor}
	d3.select(this.svg).attr("r",this.rmin).style("fill",color)
	};
	
    this.remove = function () {
	d3.select(this.svg).remove();
	delete gui.components[this.where].crumbs[this.rId];
    }

    this.where = where
    this.rId   = id
    this.svg   = "#"+this.where+'-breadcrumb-'+this.rId
    this.rmin  = 4
    this.rmax  = 6
    this.defaultColor = "white"
}
