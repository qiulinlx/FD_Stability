var years = ee.List.sequence(2010, 2019);

var scaleFactor = 0.0001; // MOD17 scale factor
var scale = 500;           // MOD17 resolution

var pts = image.sample({
  region: image.geometry(),
  scale: image.projection().nominalScale(),
  geometries: true
});

var getYearNPP = function(year) {
  year = ee.Number(year).toInt();

  var npp = ee.ImageCollection('MODIS/061/MOD17A3HGF')
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .select('Npp')
    .first()
    .multiply(0.0001);

  var sampled = npp.sampleRegions({
    collection: pts,
    scale: 500,
    geometries: true
  });

  // attach year to each feature
  return sampled.map(function(f) {
    return f.set('year', year);
  });
};

var results = ee.FeatureCollection(
  years.map(getYearNPP)
).flatten();


Export.table.toDrive({
  collection: results,
  description: 'npp_10_years',
  fileFormat: 'CSV'
});
