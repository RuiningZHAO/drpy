import sys
sys.path.append('/data3/zrn/workspace/github/drpsy/src/')

from drpsy import conf, CCDDataList

conf.unit_ccddata = 'count'

CCDDataList.read([''])