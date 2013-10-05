$(function() {
    $("#load_data").click(function() {loadData();});
    $("#colormaps").children().click(function () {changeBackground();});
    $("#scales").children().click(function () {changeBackground();});
    $("#new_region").click(function () {newRegion(svgw/2,svgh/2,newColor(uniqueId+1));});
    $("#delete").click(function () {removeSelected(); recalculatePlot();});
    $("#recalculate").click(function () {recalculatePlot();});
  });