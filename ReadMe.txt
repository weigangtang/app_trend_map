add layer of watershed when click the stream gauges
remove the previous one

- first convert shapefile to geojson
- add layer and remove

in hovertext, if glacier is "nan", it implies that watershed boundary is unavailable at that watershed.


------------------------------------------------------------------
Zoom to Select Polygon

    gdf_sel = gdf[gdf.index == sel_sid]

    gjs_sel = eval(gdf_sel.to_json())
    coords = gjs_sel['features'][0]['geometry']['coordinates'][0]
    coords = np.array(coords)

    # get centroid
    cen_pt = gdf_sel['geometry'].centroid
    cen_lon, cen_lat = cen_pt.x.values[0], cen_pt.y.values[0]

    # get zoom (rescale to watershed size)
    margin = 6.  # adjust window size
    width_to_height = 1

    min_lat, max_lat = np.min(coords[:, 1]), np.max(coords[:, 1])
    min_lon, max_lon = np.min(coords[:, 0]), np.max(coords[:, 0])

    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])

    height = (max_lat - min_lat) * margin * width_to_height
    width = (max_lon - min_lon) * margin
    lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
    lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
    zoom = round(min(lon_zoom, lat_zoom), 2)