// script to manage the loading page

var selectedProject = null;
var mouseoverAlpha = 0.2
var selectedAlpha  = 0.8

// whenever a new project is added to the analysis, update this dictionary
var colors = {'fth':[255,78,0],'cdi':[0,200,0],'xpcs':[39,114,255]}

// define a function to control the value of the rbga property
var changeColor = function(name,alpha,duration) {
    duration = duration || 100;
    var color = 'rgba('+colors[name].join(',')+','+alpha+')';
    //$('#'+name).css('background-color', color);
    d3.select('#'+name).transition().duration(duration).style('background-color',color)
}

// select the experiments, add behavior
$('#experiments div').each(function () {
    
    $(this).mouseenter(function () {
	if (selectedProject == this.id) { }
	else {changeColor(this.id,mouseoverAlpha);}}
    );
    
    $(this).click(function () {
	if (selectedProject == this.id) {
	    changeColor(this.id,0);
	    selectedProject = null;
	    $('#fileinput').attr("name","");
	    $('#readout').text("Selected experiment: None");
	}
	
	else {
	    changeColor(this.id,selectedAlpha,200);
	    if (selectedProject != null) {changeColor(selectedProject,0,300);}
	    selectedProject = this.id;
	    $("#fileinput").attr("name",this.id);
	    $('#readout').text("Selected experiment: "+this.id);
	}
    });
	
    $(this).mouseleave(function () {
	if (selectedProject == this.id) { }
	else {changeColor(this.id,0.0,500);}
    });

});

var validateForm = function () {
    
    // before submitting, we need to verify that
    // 1. a project has been selected
    // 2. a data file has been selected

    if (selectedProject == null) {
	alert("You must select an experiment first");
	return false;
    }
    
    console.log($('#fileinput').attr("name"))
    
    if ($('#fileinput').attr("name") == "") {
	alert("You must select a dataset first");
	return false;
    }
    
    return true;
}