DROP TABLE IF EXISTS ppcx_app_collapse;
CREATE TABLE IF NOT EXISTS ppcx_app_collapse (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES ppcx_app_image(id),
    geom geometry(Geometry,0),
    area DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create a view that inverts Y-axis for QGIS visualization
DROP VIEW IF EXISTS collapse_qgis;
CREATE OR REPLACE VIEW collapse_qgis AS
SELECT 
    c.id,
    c.image_id,
    i.acquisition_timestamp::date AS date,
    c.area,
    c.volume,
    -- Transform geometry: invert Y coordinates for both Polygon and MultiPolygon
    CASE 
        WHEN ST_GeometryType(c.geom) = 'ST_Polygon' THEN
            ST_SetSRID(
                ST_MakePolygon(
                    ST_MakeLine(
                        ARRAY(
                            SELECT ST_MakePoint(ST_X((dp).geom), -ST_Y((dp).geom))
                            FROM ST_DumpPoints(ST_ExteriorRing(c.geom)) AS dp
                        )
                    )
                ),
                0
            )
        WHEN ST_GeometryType(c.geom) = 'ST_MultiPolygon' THEN
            ST_SetSRID(
                ST_Collect(
                    ARRAY(
                        SELECT ST_MakePolygon(
                            ST_MakeLine(
                                ARRAY(
                                    SELECT ST_MakePoint(ST_X((dp).geom), -ST_Y((dp).geom))
                                    FROM ST_DumpPoints(ST_ExteriorRing((dumped).geom)) AS dp
                                )
                            )
                        )
                        FROM ST_Dump(c.geom) AS dumped
                    )
                ),
                0
            )
    END AS geom
FROM ppcx_app_collapse c
JOIN ppcx_app_image i ON i.id = c.image_id;