import geoip2.database
reader = geoip2.database.Reader("geo/GeoLite2-City_20261103/GeoLite2-City.mmdb")
r = reader.city("8.8.8.8")
print(r.country.name, r.city.name, r.location.latitude, r.location.longitude, r.location.accuracy_radius)