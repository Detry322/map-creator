import os, requests, subprocess, tempfile, math, glob

from PIL import Image

from app import CITIES, INPUT_TILES_FOLDER, TILE_DOWNLOADER, ZOOM_LEVEL

MAX_TILES_PER_CITY = 5000
TILE_SERVER = "OpenCycleMap"

MAP_API_BASE = 'http://maps.googleapis.com/maps/api/geocode/json'

XML_BASE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">
<properties>
  <entry key="Type">BBoxLatLon</entry>
  <entry key="OutputLocation">{}</entry>
  <entry key="TileServer">{}</entry>
  <entry key="MaxLat">{}</entry>
  <entry key="MaxLon">{}</entry>
  <entry key="MinLat">{}</entry>
  <entry key="MinLon">{}</entry>
  <entry key="OutputZoomLevel">{}</entry>
</properties>
"""

def deg2num(lat_deg, lon_deg):
  lat_rad = math.radians(float(lat_deg))
  n = 2.0 ** ZOOM_LEVEL
  xtile = int((float(lon_deg) + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def get_city_bounds(city):
  result = requests.get(MAP_API_BASE, params={'address': city})
  assert result.status_code == 200, "Maps API could not find city"
  assert len(result.json()['results']) > 0, "Maps API could not find city"
  return result.json()['results'][0]['geometry']['bounds']

def check_city_bounds(city_bounds):
  xne, yne = deg2num(city_bounds['northeast']['lat'], city_bounds['northeast']['lng'])
  xsw, ysw = deg2num(city_bounds['southwest']['lat'], city_bounds['southwest']['lng'])
  return xne > xsw and yne < ysw and (xne - xsw)*(ysw - yne) < MAX_TILES_PER_CITY

def get_xml(city_bounds):
  return XML_BASE.format(
    INPUT_TILES_FOLDER,
    TILE_SERVER,
    city_bounds['northeast']['lat'],
    city_bounds['northeast']['lng'],
    city_bounds['southwest']['lat'],
    city_bounds['southwest']['lng'],
    ZOOM_LEVEL
  )

def create_city_xml_file(city_bounds):
  fd, filename = tempfile.mkstemp()
  with os.fdopen(fd, 'w') as f:
    f.write(get_xml(city_bounds))
  return filename

def download_city(city, city_bounds):
  if not check_city_bounds(city_bounds):
    raise Exception("City's limits are invalid")
  xml_file = create_city_xml_file(city_bounds)
  subprocess.check_call(
    ['java', '-jar', TILE_DOWNLOADER, 'dl=' + xml_file]
  )
  os.unlink(xml_file)

def prune_tiles():
  glob_path = os.path.join(INPUT_TILES_FOLDER, '*', '*', '*.png')
  for filename in glob.glob(glob_path):
    try:
      Image.open(filename).verify()
    except IOError:
      print "Pruning {}...".format(filename)
      os.unlink(filename)

def download_tiles():
  for city in CITIES:
    try:
      print city
      city_bounds = get_city_bounds(city)
      download_city(city, city_bounds)
    except KeyboardInterrupt:
      raise
    except:
      print "Error downloading city: " + city

