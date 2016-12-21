import os, requests, subprocess, tempfile

from app import CITIES, INPUT_TILES_FOLDER, TILE_DOWNLOADER

ZOOM_LEVEL = "15"
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

def get_xml(city):
  result = requests.get(MAP_API_BASE, params={'address': city})
  assert result.status_code == 200, "Maps API could not find city"
  city_bounds = result.json()['results'][0]['geometry']['bounds']
  return XML_BASE.format(
    INPUT_TILES_FOLDER,
    TILE_SERVER,
    city_bounds['northeast']['lat'],
    city_bounds['northeast']['lng'],
    city_bounds['southwest']['lat'],
    city_bounds['southwest']['lng'],
    ZOOM_LEVEL
  )

def create_city_xml_file(city):
  fd, filename = tempfile.mkstemp()
  with os.fdopen(fd, 'w') as f:
    f.write(get_xml(city))
  return filename

def download_city(city):
  xml_file = create_city_xml_file(city)
  subprocess.check_call(
    ['java', '-jar', TILE_DOWNLOADER, 'dl=' + xml_file]
  )
  os.unlink(xml_file)

def download_tiles():
  for city in CITIES:
    download_city(city)
