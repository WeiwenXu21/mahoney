import json

from mahoney import io


# This is a subset of the regions.json for dataset 01.00.
# Each ROI in this example is contiguous and non-overlapping.
contiguous_nonoverlap = '''
[
  { "coordinates": [
    [326,346], [327,346], [325,347], [326,347], [327,347], [328,347], [329,347], [322,348], [323,348], [324,348], [325,348], [326,348], [327,348], [328,348], [329,348], [322,349], [323,349], [324,349], [325,349], [326,349], [327,349], [328,349], [329,349], [330,349], [331,349], [322,350], [323,350], [324,350], [325,350], [326,350], [327,350], [328,350], [329,350], [330,350], [331,350], [332,350], [321,351], [322,351], [323,351], [324,351], [325,351], [326,351], [327,351], [328,351], [329,351], [330,351], [331,351], [332,351], [333,351], [319,352], [320,352], [321,352], [322,352], [323,352], [324,352], [325,352], [326,352], [327,352], [328,352], [329,352], [330,352], [331,352], [332,352], [333,352], [319,353], [320,353], [321,353], [322,353], [323,353], [324,353], [325,353], [326,353], [327,353], [328,353], [329,353], [330,353], [331,353], [332,353], [333,353], [334,353], [319,354], [320,354], [321,354], [322,354], [323,354], [324,354], [325,354], [326,354], [327,354], [328,354], [329,354], [330,354], [331,354], [332,354], [333,354], [334,354], [320,355], [321,355], [322,355], [323,355], [324,355], [325,355], [326,355], [327,355], [328,355], [329,355], [330,355], [331,355], [332,355], [333,355], [334,355], [319,356], [320,356], [321,356], [322,356], [323,356], [324,356], [325,356], [326,356], [327,356], [328,356], [329,356], [330,356], [331,356], [332,356], [333,356], [334,356], [319,357], [320,357], [321,357], [322,357], [323,357], [324,357], [325,357], [326,357], [327,357], [328,357], [329,357], [330,357], [331,357], [332,357], [333,357], [320,358], [321,358], [322,358], [323,358], [324,358], [325,358], [326,358], [327,358], [328,358], [329,358], [330,358], [331,358], [332,358], [333,358], [320,359], [321,359], [322,359], [323,359], [324,359], [325,359], [326,359], [327,359], [328,359], [329,359], [330,359], [331,359], [332,359], [320,360], [321,360], [322,360], [323,360], [324,360], [325,360], [326,360], [327,360], [328,360], [329,360], [330,360], [331,360], [332,360], [321,361], [322,361], [323,361], [324,361], [325,361], [326,361], [327,361], [328,361], [329,361], [330,361], [331,361], [332,361], [322,362], [323,362], [324,362], [325,362], [326,362], [327,362], [328,362], [329,362], [330,362], [331,362], [327,363], [328,363], [329,363], [330,363]
  ] },
  { "coordinates": [
    [389,280], [390,280], [391,280], [387,281], [388,281], [389,281], [390,281], [391,281], [392,281], [386,282], [387,282], [388,282], [389,282], [390,282], [391,282], [392,282], [393,282], [385,283], [386,283], [387,283], [388,283], [389,283], [390,283], [391,283], [392,283], [393,283], [394,283], [385,284], [386,284], [387,284], [388,284], [389,284], [390,284], [391,284], [392,284], [393,284], [394,284], [383,285], [384,285], [385,285], [386,285], [387,285], [388,285], [389,285], [390,285], [391,285], [392,285], [393,285], [394,285], [395,285], [383,286], [384,286], [385,286], [386,286], [387,286], [388,286], [389,286], [390,286], [391,286], [392,286], [393,286], [394,286], [395,286], [383,287], [384,287], [385,287], [386,287], [387,287], [388,287], [389,287], [390,287], [391,287], [392,287], [393,287], [394,287], [395,287], [383,288], [384,288], [385,288], [386,288], [387,288], [388,288], [389,288], [390,288], [391,288], [392,288], [393,288], [394,288], [395,288], [396,288], [397,288], [385,289], [386,289], [387,289], [388,289], [389,289], [390,289], [391,289], [392,289], [393,289], [394,289], [395,289], [396,289], [397,289], [398,289], [385,290], [386,290], [387,290], [388,290], [389,290], [390,290], [391,290], [392,290], [393,290], [394,290], [395,290], [396,290], [397,290], [398,290], [385,291], [386,291], [387,291], [388,291], [389,291], [390,291], [391,291], [392,291], [393,291], [394,291], [395,291], [396,291], [397,291], [398,291], [386,292], [387,292], [388,292], [389,292], [390,292], [391,292], [392,292], [393,292], [394,292], [395,292], [396,292], [386,293], [387,293], [388,293], [389,293], [390,293], [391,293], [392,293], [393,293], [394,293], [386,294], [387,294], [388,294], [389,294], [390,294], [391,294], [392,294], [393,294], [394,294], [387,295], [388,295], [389,295], [390,295], [391,295], [392,295], [393,295], [394,295], [388,296], [389,296], [390,296], [391,296], [392,296], [393,296], [394,296], [389,297], [390,297], [391,297], [393,297]
  ] },
  { "coordinates": [
    [253,460], [251,461], [252,461], [253,461], [254,461], [249,462], [250,462], [251,462], [252,462], [253,462], [254,462], [247,463], [248,463], [249,463], [250,463], [251,463], [252,463], [253,463], [254,463], [255,463], [247,464], [248,464], [249,464], [250,464], [251,464], [252,464], [253,464], [254,464], [255,464], [256,464], [247,465], [248,465], [249,465], [250,465], [251,465], [252,465], [253,465], [254,465], [255,465], [256,465], [257,465], [247,466], [248,466], [249,466], [250,466], [251,466], [252,466], [253,466], [254,466], [255,466], [256,466], [257,466], [258,466], [259,466], [247,467], [248,467], [249,467], [250,467], [251,467], [252,467], [253,467], [254,467], [255,467], [256,467], [257,467], [258,467], [259,467], [247,468], [248,468], [249,468], [250,468], [251,468], [252,468], [253,468], [254,468], [255,468], [256,468], [257,468], [258,468], [259,468], [260,468], [248,469], [249,469], [250,469], [251,469], [252,469], [253,469], [254,469], [255,469], [256,469], [257,469], [258,469], [259,469], [260,469], [248,470], [249,470], [250,470], [251,470], [252,470], [253,470], [254,470], [255,470], [256,470], [257,470], [258,470], [259,470], [260,470], [248,471], [249,471], [250,471], [251,471], [252,471], [253,471], [254,471], [255,471], [256,471], [257,471], [258,471], [259,471], [260,471], [248,472], [249,472], [250,472], [251,472], [252,472], [253,472], [254,472], [255,472], [256,472], [257,472], [258,472], [259,472], [260,472], [249,473], [250,473], [251,473], [252,473], [253,473], [254,473], [255,473], [256,473], [257,473], [258,473], [259,473], [260,473], [249,474], [250,474], [251,474], [252,474], [253,474], [254,474], [255,474], [256,474], [257,474], [258,474], [259,474], [260,474], [249,475], [250,475], [251,475], [252,475], [253,475], [254,475], [255,475], [256,475], [257,475], [258,475], [259,475], [260,475], [248,476], [249,476], [250,476], [251,476], [252,476], [253,476], [254,476], [255,476], [256,476], [257,476], [258,476], [259,476], [260,476], [250,477], [251,477], [252,477], [253,477], [254,477], [255,477], [256,477], [257,477], [258,477], [259,477], [260,477], [254,478], [255,478], [256,478], [257,478], [258,478], [259,478], [260,478], [255,479], [256,479]
  ] },
  { "coordinates": [
    [164,239], [165,239], [166,239], [167,239], [168,239], [165,240], [166,240], [167,240], [168,240], [165,241], [166,241], [167,241], [168,241], [169,241], [170,241], [171,241], [172,241], [173,241], [174,241], [165,242], [166,242], [167,242], [168,242], [169,242], [170,242], [171,242], [172,242], [173,242], [174,242], [175,242], [176,242], [177,242], [165,243], [166,243], [167,243], [168,243], [169,243], [170,243], [171,243], [172,243], [173,243], [174,243], [175,243], [176,243], [177,243], [165,244], [166,244], [167,244], [168,244], [169,244], [170,244], [171,244], [172,244], [173,244], [174,244], [175,244], [176,244], [177,244], [165,245], [166,245], [167,245], [168,245], [169,245], [170,245], [171,245], [172,245], [173,245], [174,245], [175,245], [176,245], [177,245], [165,246], [166,246], [167,246], [168,246], [169,246], [170,246], [171,246], [172,246], [173,246], [174,246], [175,246], [176,246], [177,246], [166,247], [167,247], [168,247], [169,247], [170,247], [171,247], [172,247], [173,247], [174,247], [175,247], [176,247], [177,247], [166,248], [167,248], [168,248], [169,248], [170,248], [171,248], [172,248], [173,248], [174,248], [175,248], [176,248], [166,249], [167,249], [168,249], [169,249], [170,249], [171,249], [172,249], [173,249], [174,249], [175,249], [176,249], [166,250], [167,250], [168,250], [169,250], [170,250], [171,250], [172,250], [173,250], [174,250], [175,250], [176,250], [166,251], [167,251], [168,251], [169,251], [170,251], [171,251], [172,251], [173,251], [174,251], [175,251], [166,252], [167,252], [168,252], [169,252], [170,252], [171,252], [166,253], [167,253]
  ] }
]
'''


def test_lossless():
    '''Converting between a segmentation mask and an ROI list should be
    lossless when the regions are contiguous and non-overlapping.
    '''
    true_rois = json.loads(contiguous_nonoverlap)
    mask = io.rois_to_mask(true_rois)
    processed_rois = io.mask_to_rois(mask)

    # There are 4 ROIs in the example data
    assert len(true_rois) == 4
    assert len(processed_rois) == 4

    # The structures should be the same modulo some ordering
    for rois in [true_rois, processed_rois]:
        for i, roi in enumerate(rois):
            rois[i] = sorted(roi['coordinates'])
        rois.sort()
    assert processed_rois == true_rois
