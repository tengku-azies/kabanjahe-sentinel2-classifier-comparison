/**** 01_gee_classification.js
 * Revisi skrip GEE Kabanjahe yang TIDAK membutuhkan data baru.
 * Memakai area, periode, filter awan, dan poligon sampel yang sama dengan skrip lama
 * (variabel import: geometry, areaterbangun, Hutan, lahanpertanianbervegetasi,
 * lahanpertaniantidakbervegetasi).
 *
 * Perbaikan dibanding skrip lama:
 *  1. Ketujuh algoritma dijalankan di GEE (satu platform), termasuk MLC, MD, dan SAM.
 *  2. Validasi silang 5-lipat dengan DUA desain pada sampel yang sama:
 *       - pixel   : piksel diacak (setara desain lama 70:30)
 *       - polygon : seluruh piksel satu poligon berada di lipatan yang sama
 *     sehingga inflasi akurasi akibat autokorelasi spasial bisa diukur.
 *  3. Naive Bayes diganti Gaussian NB. ee.Classifier.smileNaiveBayes mengasumsikan
 *     fitur bilangan bulat positif (data cacah), tidak cocok untuk reflektansi.
 *  4. SVM memakai kernel RBF pada band terstandardisasi (skrip lama: kernel linear
 *     dengan nilai mentah 0-10000).
 *  5. Opsional: surface reflectance (L2A) dan masking awan Cloud Score+ pada citra
 *     yang SAMA (filter tanggal dan persentase awan tidak diubah).
 *  6. Semua sampling, luas, dan ekspor memakai EPSG:32647 10 m. Skrip lama mengekspor
 *     peta LULC tanpa crs (jadi EPSG:4326), penyebab selisih luas GEE vs QGIS di naskah.
 *  7. Prediksi per piksel (out-of-fold) diekspor, sehingga uji McNemar 21 pasangan,
 *     selang kepercayaan per poligon, dan inflasi bisa dihitung dengan 03_stats_from_gee.py.
 *
 * CATATAN: skrip ini belum pernah dijalankan (saya tidak punya akses ke akun dan aset
 * Anda). Jika ada error, salin pesan error dari Console dan kirimkan ke saya.
 ****/

// ============================== PENGATURAN ==============================
var area = geometry;
var startDate = '2024-07-01';
var endDate = '2026-06-30';
var MAX_CLOUD = 10;          // filter yang sama dengan skrip lama

var USE_SR = true;           // true = Level-2A (surface reflectance), false = TOA seperti skrip lama
var USE_CLOUD_MASK = true;   // masking awan per piksel dengan Cloud Score+
var CS_THRESHOLD = 0.60;

var N_FOLDS = 5;
var SEED = 42;
var CRS = 'EPSG:32647';
var SCALE = 10;
var FOLDER = 'GEE_Revisi';

// Hiperparameter. Isi dengan hasil RUN_TUNING bila tuning dijalankan.
var PARAMS = {
  RF:   {numberOfTrees: 500, variablesPerSplit: 3, minLeafPopulation: 1},
  CART: {minLeafPopulation: 5},
  SVM:  {cost: 100, gamma: 0.1}
};
var RUN_TUNING = false;      // true = ekspor tabel akurasi grid hiperparameter (CV per poligon)

var BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
var Z_BANDS = BANDS.map(function (b) { return b + '_z'; });
var NB = BANDS.length;
var CLASS_NAMES = ['Area Terbangun', 'Hutan', 'Pertanian Bervegetasi', 'Pertanian Tidak Bervegetasi'];
var CLASS_CODES = [0, 1, 2, 3];
var PALETTE = ['D73000', '1A7F37', 'FFFF66', 'C8A165'];   // terbangun merah, hutan hijau
var METHODS = ['RF', 'CART', 'SVM', 'NB', 'MLC', 'MD', 'SAM'];
// ========================================================================


// ---------------------------- 1. KOMPOSIT ----------------------------
var colId = USE_SR ? 'COPERNICUS/S2_SR_HARMONIZED' : 'COPERNICUS/S2_HARMONIZED';
var S2 = ee.ImageCollection(colId)
  .filterBounds(area)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD));

if (USE_CLOUD_MASK) {
  var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
  S2 = S2.linkCollection(csPlus, ['cs_cdf']).map(function (img) {
    return img.updateMask(img.select('cs_cdf').gte(CS_THRESHOLD));
  });
}

var composite = S2.select(BANDS).median().clip(area);

print('Koleksi:', colId);
print('Jumlah citra:', S2.size());
print('Tanggal citra:', S2.aggregate_array('system:time_start')
  .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }));

// Standardisasi untuk SVM: statistik dari seluruh piksel kecamatan (tanpa label,
// sehingga tidak membocorkan informasi data uji).
var st = composite.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
  geometry: area, scale: SCALE, crs: CRS, maxPixels: 1e13, tileScale: 4
});
var zImage = ee.Image.cat(BANDS.map(function (b) {
  return composite.select(b)
    .subtract(ee.Number(st.get(b + '_mean')))
    .divide(ee.Number(st.get(b + '_stdDev')))
    .rename(b + '_z');
}));
var stack = composite.addBands(zImage);

Map.centerObject(area, 13);
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 300, max: 2500}, 'Komposit');


// ---------------------- 2. POLIGON SAMPEL + LIPATAN ----------------------
// Pecah setiap kelas menjadi poligon tunggal (poligon yang bersentuhan digabung,
// karena piksel di kedua sisinya tidak independen), beri poly_id, lalu bagi poligon
// ke 5 lipatan secara bergiliran dalam tiap kelas (stratified).
var SRC = [areaterbangun, Hutan, lahanpertanianbervegetasi, lahanpertaniantidakbervegetasi];

function polygonsOf(fc, code) {
  var parts = ee.FeatureCollection(fc).geometry().geometries();
  var polys = ee.FeatureCollection(parts.map(function (g) {
    return ee.Feature(ee.Geometry(g));
  })).filter(ee.Filter.hasType('.geo', 'Polygon'))
    .randomColumn('r', SEED).sort('r');
  var lst = polys.toList(polys.size());
  return ee.FeatureCollection(ee.List.sequence(0, polys.size().subtract(1)).map(function (i) {
    i = ee.Number(i);
    return ee.Feature(lst.get(i)).set({
      'class': code,
      'poly_id': ee.Number(code).multiply(100000).add(i),
      'fold_poly': i.mod(N_FOLDS)
    });
  }));
}

var polys = ee.FeatureCollection(CLASS_CODES.map(function (c) {
  return polygonsOf(SRC[c], c);
})).flatten();

print('Jumlah poligon per kelas (0=BU,1=F,2=VC,3=NVC). Minimal 5, idealnya >= 30:',
  polys.aggregate_histogram('class'));
Map.addLayer(polys, {color: 'yellow'}, 'Poligon sampel', false);


// ------------------------- 3. EKSTRAKSI PIKSEL -------------------------
var samples = stack.addBands(ee.Image.pixelLonLat()).sampleRegions({
  collection: polys,
  properties: ['class', 'poly_id', 'fold_poly'],
  scale: SCALE,
  projection: CRS,
  tileScale: 4,
  geometries: false
}).randomColumn('rp', SEED + 1).map(function (f) {
  return f.set('fold_pix', ee.Number(f.get('rp')).multiply(N_FOLDS).floor());
});

print('Jumlah piksel sampel:', samples.size());
print('Piksel per kelas:', samples.aggregate_histogram('class'));


// ------------------------- 4. DEFINISI ALGORITMA -------------------------
// Algoritma bawaan GEE. SVM memakai band terstandardisasi; algoritma lain memakai
// nilai reflektansi asli (SAM dan MD bergantung pada skala spektral asli).
function builtins(p) {
  return {
    RF:   {cl: ee.Classifier.smileRandomForest({numberOfTrees: p.RF.numberOfTrees,
             variablesPerSplit: p.RF.variablesPerSplit,
             minLeafPopulation: p.RF.minLeafPopulation, seed: SEED}), inp: BANDS},
    CART: {cl: ee.Classifier.smileCart({minLeafPopulation: p.CART.minLeafPopulation}), inp: BANDS},
    SVM:  {cl: ee.Classifier.libsvm({svmType: 'C_SVC', kernelType: 'RBF',
             cost: p.SVM.cost, gamma: p.SVM.gamma}), inp: Z_BANDS},
    MD:   {cl: ee.Classifier.minimumDistance('euclidean'), inp: BANDS},
    // 'cosine' = sudut spektral terhadap rerata kelas, yaitu Spectral Angle Mapper.
    SAM:  {cl: ee.Classifier.minimumDistance('cosine'), inp: BANDS}
  };
}

// MLC (kovarians penuh) dan Gaussian Naive Bayes (kovarians diagonal), prior sama.
// Fungsi diskriminan: d_k(x) = ln|S_k| + (x - m_k)' S_k^-1 (x - m_k); kelas = argmin d_k.
function fitGaussian(train, diagonal) {
  var I = ee.Array.identity(NB);
  return CLASS_CODES.map(function (c) {
    var fc = train.filter(ee.Filter.eq('class', c));
    // Matriks data NB x n, lalu rerata dan kovarians dihitung dengan operasi array
    // (menghindari ee.Reducer.covariance yang menolak properti array tanpa tipe 1-D).
    var data = ee.Array(ee.List(fc.reduceColumns(ee.Reducer.toList().repeat(NB), BANDS)
      .get('list')));
    var n = ee.Number(data.length().get([1]));
    var mu = data.reduce(ee.Reducer.mean(), [1]);                 // NB x 1
    var centered = data.subtract(mu.repeat(1, n));
    var cov = centered.matrixMultiply(centered.transpose()).divide(n.subtract(1));
    if (diagonal) { cov = cov.multiply(I); }
    // Regularisasi kecil agar matriks selalu dapat diinvers.
    var ridge = cov.multiply(I).reduce(ee.Reducer.sum(), [0, 1]).get([0, 0])
      .divide(NB).multiply(1e-6);
    cov = cov.add(I.multiply(ridge));
    var L = ee.Array(ee.Dictionary(cov.matrixCholeskyDecomposition()).get('L'));
    var logdet = L.matrixDiagonal().log().reduce(ee.Reducer.sum(), [0]).get([0, 0]).multiply(2);
    return {mu: mu, inv: cov.matrixInverse(), logdet: logdet};
  });
}

function gaussClassifyFC(fc, model, outName) {
  return fc.map(function (f) {
    var x = ee.Array(f.toDictionary(BANDS).values(BANDS)).reshape([NB, 1]);
    var d = ee.List(model.map(function (m) {
      var diff = x.subtract(m.mu);
      return diff.transpose().matrixMultiply(m.inv).matrixMultiply(diff)
        .get([0, 0]).add(m.logdet);
    }));
    return f.set(outName, d.indexOf(d.reduce(ee.Reducer.min())));
  });
}

function gaussClassifyImage(img, model) {
  var negD = model.map(function (m) {
    var diffs = BANDS.map(function (b, i) {
      return img.select(b).subtract(m.mu.get([i, 0]));
    });
    var q = ee.Image(0);
    for (var i = 0; i < NB; i++) {
      for (var j = i; j < NB; j++) {
        var w = m.inv.get([i, j]);
        if (j > i) { w = ee.Number(w).multiply(2); }
        q = q.add(diffs[i].multiply(diffs[j]).multiply(w));
      }
    }
    return q.add(m.logdet).multiply(-1);
  });
  return ee.Image.cat(negD).toArray().arrayArgmax().arrayGet([0]).toByte();
}

// Latih ketujuh algoritma dan klasifikasikan satu FeatureCollection uji.
function classifyAll(train, test, p) {
  var b = builtins(p);
  ['RF', 'CART', 'SVM', 'MD', 'SAM'].forEach(function (m) {
    var model = b[m].cl.train({features: train, classProperty: 'class', inputProperties: b[m].inp});
    test = test.classify(model, m);
  });
  test = gaussClassifyFC(test, fitGaussian(train, true), 'NB');
  test = gaussClassifyFC(test, fitGaussian(train, false), 'MLC');
  return test;
}


// ------------------- 5. VALIDASI SILANG (DUA DESAIN) -------------------
function crossValidate(foldField, label) {
  var parts = [];
  for (var k = 0; k < N_FOLDS; k++) {
    var train = samples.filter(ee.Filter.neq(foldField, k));
    var test = samples.filter(ee.Filter.eq(foldField, k));
    parts.push(classifyAll(train, test, PARAMS).map(function (f) {
      return f.set({fold: k, design: label});
    }));
  }
  return ee.FeatureCollection(parts).flatten();
}

var oofPixel = crossValidate('fold_pix', 'pixel');
var oofPoly = crossValidate('fold_poly', 'polygon');

var OOF_COLS = ['design', 'fold', 'class', 'poly_id'].concat(METHODS);
Export.table.toDrive({collection: oofPixel, description: 'oof_pixel', folder: FOLDER,
  fileNamePrefix: 'oof_pixel', fileFormat: 'CSV', selectors: OOF_COLS});
Export.table.toDrive({collection: oofPoly, description: 'oof_polygon', folder: FOLDER,
  fileNamePrefix: 'oof_polygon', fileFormat: 'CSV', selectors: OOF_COLS});

// Ringkasan cepat OA dan kappa (tabel lengkap dan uji statistik: 03_stats_from_gee.py).
function summary(oof, label) {
  return ee.FeatureCollection(METHODS.map(function (m) {
    var em = oof.errorMatrix('class', m, CLASS_CODES);
    return ee.Feature(null, {design: label, model: m,
      OA: ee.Number(em.accuracy()).multiply(100), kappa: em.kappa()});
  }));
}
Export.table.toDrive({
  collection: summary(oofPixel, 'pixel').merge(summary(oofPoly, 'polygon')),
  description: 'ringkasan_OA', folder: FOLDER, fileNamePrefix: 'ringkasan_OA',
  fileFormat: 'CSV', selectors: ['design', 'model', 'OA', 'kappa']
});


// ------------------- 6. PETA DAN LUAS (MODEL AKHIR) -------------------
// Model akhir dilatih dengan SELURUH sampel. Luas dihitung dari jumlah piksel peta
// (luas terpetakan, belum terkoreksi galat).
var bFinal = builtins(PARAMS);
var maps = {};
['RF', 'CART', 'SVM', 'MD', 'SAM'].forEach(function (m) {
  var model = bFinal[m].cl.train({features: samples, classProperty: 'class',
    inputProperties: bFinal[m].inp});
  maps[m] = stack.classify(model).toByte();
});
maps.NB = gaussClassifyImage(stack, fitGaussian(samples, true));
maps.MLC = gaussClassifyImage(stack, fitGaussian(samples, false));

var areaRows = [];
METHODS.forEach(function (m, i) {
  var cls = maps[m].clip(area).rename('class');
  Map.addLayer(cls, {min: 0, max: 3, palette: PALETTE}, 'Klasifikasi ' + m, i === 0);

  var groups = ee.List(ee.Image.pixelArea().divide(10000).addBands(cls).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: area, scale: SCALE, crs: CRS, maxPixels: 1e13, tileScale: 4
  }).get('groups'));
  areaRows.push(groups.map(function (g) {
    g = ee.Dictionary(g);
    var k = ee.Number(g.get('class')).toInt();
    return ee.Feature(null, {Algoritma: m, Kelas: ee.List(CLASS_NAMES).get(k),
      Luas_ha: ee.Number(g.get('sum'))});
  }));

  Export.image.toDrive({image: cls, description: 'LULC_' + m, folder: FOLDER,
    fileNamePrefix: 'LULC_' + m, region: area, scale: SCALE, crs: CRS, maxPixels: 1e13});
});

Export.table.toDrive({
  collection: ee.FeatureCollection(ee.List(areaRows).flatten()),
  description: 'luas_terpetakan', folder: FOLDER, fileNamePrefix: 'luas_terpetakan',
  fileFormat: 'CSV', selectors: ['Algoritma', 'Kelas', 'Luas_ha']
});


// ------------------- 7. EKSPOR DATA UNTUK PYTHON -------------------
// Dipakai 02_reanalysis.py (tuning bersarang yang lebih ketat) bila diperlukan.
Export.table.toDrive({
  collection: samples, description: 'samples', folder: FOLDER, fileNamePrefix: 'samples',
  fileFormat: 'CSV', selectors: BANDS.concat(['class', 'poly_id', 'longitude', 'latitude'])
});
Export.image.toDrive({
  image: composite.toUint16(), description: 'composite_10band', folder: FOLDER,
  fileNamePrefix: 'composite_10band', region: area, scale: SCALE, crs: CRS, maxPixels: 1e13
});
Export.table.toDrive({collection: polys, description: 'poligon_sampel', folder: FOLDER,
  fileNamePrefix: 'poligon_sampel', fileFormat: 'SHP'});


// ------------------- 8. TUNING HIPERPARAMETER (OPSIONAL) -------------------
// CV per poligon untuk setiap kombinasi. Pilih kombinasi dengan OA tertinggi, isi ke
// PARAMS, lalu jalankan ulang. Catatan untuk naskah: tuning memakai lipatan yang sama
// dengan evaluasi (tidak bersarang), sehingga sedikit optimistis untuk RF/CART/SVM.
// Versi bersarang yang ketat tersedia di 02_reanalysis.py.
if (RUN_TUNING) {
  var grid = [];
  [1, 10, 100, 1000].forEach(function (C) {
    [0.01, 0.1, 1].forEach(function (g) {
      grid.push({model: 'SVM', param: 'C=' + C + ', gamma=' + g, inp: Z_BANDS,
        cl: ee.Classifier.libsvm({svmType: 'C_SVC', kernelType: 'RBF', cost: C, gamma: g})});
    });
  });
  [2, 3, 5].forEach(function (v) {
    [1, 5].forEach(function (leaf) {
      grid.push({model: 'RF', param: 'vps=' + v + ', minLeaf=' + leaf, inp: BANDS,
        cl: ee.Classifier.smileRandomForest({numberOfTrees: 300, variablesPerSplit: v,
          minLeafPopulation: leaf, seed: SEED})});
    });
  });
  [1, 5, 10, 20].forEach(function (leaf) {
    grid.push({model: 'CART', param: 'minLeaf=' + leaf, inp: BANDS,
      cl: ee.Classifier.smileCart({minLeafPopulation: leaf})});
  });

  var tuneRows = grid.map(function (g) {
    var correct = ee.Number(0);
    for (var k = 0; k < N_FOLDS; k++) {
      var tr = samples.filter(ee.Filter.neq('fold_poly', k));
      var te = samples.filter(ee.Filter.eq('fold_poly', k));
      var pred = te.classify(g.cl.train({features: tr, classProperty: 'class',
        inputProperties: g.inp}), 'p');
      correct = correct.add(pred.filter(ee.Filter.equals({leftField: 'class',
        rightField: 'p'})).size());
    }
    return ee.Feature(null, {model: g.model, param: g.param,
      OA: correct.divide(samples.size()).multiply(100)});
  });
  Export.table.toDrive({collection: ee.FeatureCollection(tuneRows), description: 'tuning',
    folder: FOLDER, fileNamePrefix: 'tuning', fileFormat: 'CSV',
    selectors: ['model', 'param', 'OA']});
}
