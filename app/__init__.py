import os

ZOOM_LEVEL = 15

BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INPUT_TILES_FOLDER = os.path.join(BASE_FOLDER, 'input_tiles')
GENERATED_TILES_FOLDER = os.path.join(BASE_FOLDER, 'generated_tiles')
DEPENDENCY_FOLDER = os.path.join(BASE_FOLDER, 'deps')

TILE_DOWNLOADER = os.path.join(DEPENDENCY_FOLDER, 'jTileDownloader-0-6-1.jar')

CITIES = [
  "Bangkok, Thailand",
  "Barcelona, Spain",
  "Berlin, Germany",
  "Bogota, Colombia",
  "Boston, Massachusetts",
  "Buenos Aires, Argentina",
  "Cairo, Egypt",
  "Chicago, IL",
  "Delhi, India",
  "Dhaka, Bangladesh",
  "Hanoi, Vietnam",
  "Ho Chi Minh City, Vietnam",
  "Hong Kong",
  "Houston, Texas",
  "Istanbul, Turkey",
  "Kolkata, India",
  "Kuala Lumpur, Malaysia",
  "Kyoto, Japan",
  "Lagos, Nigeria",
  "Las Vegas, Nevada",
  "Lima, Peru",
  "London, England",
  "Los Angeles, California",
  "Madrid, Spain",
  "Manila, Philippines",
  "Melbourne, Australia",
  "Mexico City, Mexico",
  "Milan, Italy",
  "Montreal, Canada",
  "Moscow, Russia",
  "Mumbai, India",
  "New Orleans, Louisiana",
  "New York, New York",
  "Osaka, Japan",
  "Paris, France",
  "Philadelphia, Pennsylvania",
  "Phoenix, Arizona",
  "Rio de Janeiro, Brazil",
  "San Antonio, Texas",
  "San Diego, California",
  "San Francisco, California",
  "Santiago, Chile",
  "Sao Paulo, Brazil",
  "Seattle, Washington",
  "Seoul, South Korea",
  "Shanghai, China",
  "Singapore, Singapore",
  "Taipei, Taiwan",
  "Tokyo, Japan",
  "Toronto, Canada",
  "Washington D.C.",
  "Wuhan, China",
]
