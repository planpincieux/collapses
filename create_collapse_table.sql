DROP TABLE IF EXISTS ppcx_app_collapse CASCADE;
CREATE TABLE IF NOT EXISTS ppcx_app_collapse (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES ppcx_app_image(id),
    geom geometry(MultiPolygon, 0),
    geom_qgis geometry(MultiPolygon, 0),
    area DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Function to invert Y coordinates of a geometry
CREATE OR REPLACE FUNCTION invert_y_coordinates(geom geometry)
RETURNS geometry AS $$
BEGIN
    IF geom IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN CASE 
        WHEN ST_GeometryType(geom) = 'ST_Polygon' THEN
            ST_Multi(  -- Wrap single polygon in MultiPolygon
                ST_SetSRID(
                    ST_MakePolygon(
                        ST_MakeLine(
                            ARRAY(
                                SELECT ST_MakePoint(ST_X((dp).geom), -ST_Y((dp).geom))
                                FROM ST_DumpPoints(ST_ExteriorRing(geom)) AS dp
                            )
                        )
                    ),
                    ST_SRID(geom)
                )
            )
        WHEN ST_GeometryType(geom) = 'ST_MultiPolygon' THEN
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
                        FROM ST_Dump(geom) AS dumped
                    )
                ),
                ST_SRID(geom)
            )
        ELSE
            geom  -- Return unchanged for other geometry types
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger function to sync geom -> geom_qgis
CREATE OR REPLACE FUNCTION sync_geom_to_qgis()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom_qgis := invert_y_coordinates(NEW.geom);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger function to sync geom_qgis -> geom
CREATE OR REPLACE FUNCTION sync_qgis_to_geom()
RETURNS TRIGGER AS $$
BEGIN
    -- Only sync if geom_qgis changed but geom didn't
    IF NEW.geom_qgis IS DISTINCT FROM OLD.geom_qgis 
       AND NEW.geom IS NOT DISTINCT FROM OLD.geom THEN
        NEW.geom := invert_y_coordinates(NEW.geom_qgis);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: When geom is inserted or updated, auto-update geom_qgis
CREATE TRIGGER trigger_sync_geom_to_qgis
    BEFORE INSERT OR UPDATE OF geom
    ON ppcx_app_collapse
    FOR EACH ROW
    EXECUTE FUNCTION sync_geom_to_qgis();

-- Trigger: When geom_qgis is updated (e.g., in QGIS), sync back to geom
CREATE TRIGGER trigger_sync_qgis_to_geom
    BEFORE UPDATE OF geom_qgis
    ON ppcx_app_collapse
    FOR EACH ROW
    EXECUTE FUNCTION sync_qgis_to_geom();

-- Optional: Create view for backward compatibility
DROP VIEW IF EXISTS collapse_qgis;
CREATE OR REPLACE VIEW collapse_qgis AS
SELECT 
    c.id,
    c.image_id,
    i.acquisition_timestamp::date AS date,
    c.area,
    c.volume,
    c.geom_qgis AS geom  -- Use the QGIS-compatible geometry
FROM ppcx_app_collapse c
JOIN ppcx_app_image i ON i.id = c.image_id;

-- Add comments for documentation
COMMENT ON COLUMN ppcx_app_collapse.geom IS 'Original geometry in image coordinates (Y-axis down) - MultiPolygon';
COMMENT ON COLUMN ppcx_app_collapse.geom_qgis IS 'Y-inverted geometry for QGIS visualization (Y-axis up) - MultiPolygon';
COMMENT ON FUNCTION invert_y_coordinates(geometry) IS 'Inverts Y coordinates of a polygon or multipolygon geometry';
COMMENT ON TRIGGER trigger_sync_geom_to_qgis ON ppcx_app_collapse IS 'Automatically updates geom_qgis when geom changes';
COMMENT ON TRIGGER trigger_sync_qgis_to_geom ON ppcx_app_collapse IS 'Automatically updates geom when geom_qgis is edited in QGIS';