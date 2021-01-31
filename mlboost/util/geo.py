from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import logging

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    verified (double checked): http://www.nhc.noaa.gov/gccalc.shtml
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def convert_time(string, time_format='tstime'):
    ''' util convert time fct '''
    if time_format == 'none':
        return lambda string:string
    # ex: 20140103020708930000
    elif time_format =='tstime':    
        timestr=string[:14]
        if not timestr.isdigit() or " " in timestr:
            logging.warning("<%s> not in appropriate tstime format" %s)
            return
        else:
            try:
                return datetime.strptime(timestr,"%Y%m%d%H%M%S")
            except Exception as ex:
                logging.warning(ex)
                return 

def speed(lon1, lat1, lon2, lat2, t1, t2, min_val=.5):
    ''' return speed in km/h '''
    if None in (lon1, lat1, lon2, lat2):
        logging.warning("Can't evaluate speed")
        return 0
    delta_t = t2 -t1
    hours = delta_t.total_seconds()/(60*60)
    if hours>0:
        kmh = haversine(lon1, lat1, lon2, lat2)/hours
        return kmh if kmh>min_val else 0
    else:
        return 0
    
def get_long_lat(string, method='<>'):
    " extract long from : <+39.2311228, -96.4469993>"
    if string[0]!='<' or string[-1]!='>' or string=="N/A":
        return None,None
    else:
        mid=string.index(',')
        return float(string[1:mid]), float(string[mid+2:-1])

class Location:
    ''' interface to geopy to retrieve location information
        leveraging https://pypi.python.org/pypi/geopy/1.9.1 '''
    def __init__(self):
        from geopy.geocoders import Nominatim
        self.geolocator = Nominatim()
        
    def reverse(self, lon, lat, timeout=100):
        ''' return location object
            ex:
            address -> location.address
            city -> location.raw['address']['city']
        '''
        try:
            return  self.geolocator.reverse("%s, %s" %(lon,lat), 
                                            timeout=timeout)
        except:
            return {}


if __name__ == "__main__":
    lon1 = 44.8684658
    lat1 = -93.3490480
    lon2 = 34.7176319
    lat2 = -89.9641628 
    t1 =  convert_time('20150101000009286000')
    t2 =  convert_time('20150101000016759000')
    # check specific user speed
    t3 = convert_time('20150215052202650000')
    t4 = convert_time('20150215052213581000')
    lon3, lat3 = get_long_lat('<+37.5123474, -121.9836775>')
    lon4, lat4 = get_long_lat('<+37.4309359, -120.7853906>')
    print("######")
    print(haversine(lon3, lat3, lon4, lat4), "km")
    print(t3, t4)
    print(speed(lon3, lat3, lon4, lat4, t3, t4), "km/h")
    print("#####")

    for string in ["<+36.7557823, -108.1926184>","<+39.2311228, -96.4469993>"]:
        print(string,"->",get_long_lat(string))
    

    print(haversine(lon1, lat1, lon2, lat2), "km")
    print(speed(lon1, lat1, lon2, lat2, t1, t2), "km/h")
    loc = Location()
    print(loc.reverse(lon1, lat1))
    import sys
    sys.stdin.readline()
    for line in sys.stdin.readlines():
        line = line.strip()
        lon, lat = get_long_lat(line.strip())
        location=loc.reverse(lon, lat) if lon!=None else "?"
        print(line, location)
