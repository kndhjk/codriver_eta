# 1) 预处理
docker run --rm -v %cd%/roadnet:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/beijing.osm.pbf
docker run --rm -v %cd%/roadnet:/data osrm/osrm-backend osrm-partition /data/beijing.osrm
docker run --rm -v %cd%/roadnet:/data osrm/osrm-backend osrm-customize /data/beijing.osrm

# 2) 启动路由服务（含 /match）
docker run -p 5000:5000 -v %cd%/roadnet:/data osrm/osrm-backend osrm-routed --algorithm mld /data/beijing.osrm
