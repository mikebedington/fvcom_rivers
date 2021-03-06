import numpy as np
import requests as rqs
import datetime as dt

EA_API='http://environment.data.gov.uk/hydrology/id/'


def parse_json_timeseries(ts):
    out_dt = []
    out_val = []
    for this_entry in ts:
        out_dt.append(dt.datetime.strptime(this_entry['dateTime'], '%Y-%m-%dT%H:%M:%SZ'))
        try:
            out_val.append(this_entry['value'])
        except KeyError:
            out_val.append(np.NaN)

    return np.asarray(out_dt), np.asarray(out_val)


class EAretrieval():
    def __init__(self, lon, lat, dist_thresh=3):
        self.lon = lon
        self.lat = lat
        self.dist_thresh = dist_thresh
        self._get_EA_station_str()

    def _get_EA_station_str(self):
        r = rqs.get('{}stations.json'.format(EA_API), params={'long':self.lon, 'lat':self.lat, 'dist':self.dist_thresh}) 
        self.station = r.json()['items'][0]['notation']
        r = rqs.get('{}measures.json'.format(EA_API), params={'station':self.station})
        
        self.readings_apis = {}
        for this_reading in r.json()['items']:
            self.readings_apis[this_reading['observationType']['label']] = this_reading['@id']

    def get_readings(self, reading_str, start_date=None):
        if start_date is not None:
            r = rqs.get(self.readings_apis[reading_str] + '/readings', params={'mineq-date':start_date.strftime('%Y-%m-%d')})
        else:
            r = rqs.get(self.readings_apis[reading_str] + '/readings')

        return parse_json_timeseries(r.json()['items'])
        
        


    
 
