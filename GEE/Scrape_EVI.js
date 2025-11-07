//--------------------------------------
// LOAD POINT TABLE
//--------------------------------------
var pts = table; // TABLE IS UPLOADED PID_LOCATION.CSV FILE ASSET

// MODIS MOD13Q1 (EVI)
var modis = ee.ImageCollection("MODIS/006/MOD13Q1");

//--------------------------------------
// YEARS
//--------------------------------------
var years = ee.List.sequence(2013, 2022);

//--------------------------------------
// SUMMER MEAN PER YEAR
//--------------------------------------
var summerByYear = years.map(function(y) {
  
  var start = ee.Date.fromYMD(y, 6, 1);
  var end   = ee.Date.fromYMD(y, 8, 31);

  var summer = modis
    .filterDate(start, end)
    .select("EVI")
    .map(function(img){
      // scale factor
      return img.multiply(0.0001).copyProperties(img, img.propertyNames());
    });

  var summerMean = summer.mean()
    .set("year", y);

  return summerMean;
});

var summerCollection = ee.ImageCollection(summerByYear);

//--------------------------------------
// SAMPLE SUMMER EVI AT POINTS
//--------------------------------------
var sampled = summerCollection.map(function(img) {
  return img.sampleRegions({
    collection: pts,
    scale: 250,
    geometries: true
  }).map(function(ft) {
    return ft.set("year", img.get("year"));
  });
}).flatten();

//--------------------------------------
// EXPORT
//--------------------------------------
Export.table.toDrive({
  collection: sampled,
  description: "Summer_EVI_2013_2022",
  fileFormat: "CSV"
});
