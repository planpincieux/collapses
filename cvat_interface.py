import logging

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from sqlalchemy import create_engine

from ppcluster.utils import ConfigManager

logger = logging.getLogger("ppcx")

config = ConfigManager()
db_engine = create_engine(config.db_url)

cvat_host = "http://150.145.51.193"
cvat_port = 8081
cvat_user = "francescoioli"
cvat_password = "vkq0hex_qah@TZA4hbm"

with make_client(
    host=cvat_host,
    port=cvat_port,
    credentials=(cvat_user, cvat_password),
) as client:
    # Let's set the organization slug to "ppcx".
    client.organization_slug = "ppcx"

    #  get the project
    proj_spec = {
        "name": "sandbox",
        "labels": [
            {
                "name": "A",
                "id": 1,
                "color": "#ff355e",
                "type": "polygon",
                "attributes": [],
            },
            {
                "name": "B",
                "id": 2,
                "color": "#f59331",
                "type": "polygon",
                "attributes": [],
            },
            {
                "name": "C",
                "id": 3,
                "color": "#fafa37",
                "type": "polygon",
                "attributes": [],
            },
            {
                "name": "D",
                "id": 4,
                "color": "#3df53d",
                "type": "polygon",
                "attributes": [],
            },
            {
                "name": "ROI",
                "id": 5,
                "color": "#32b7fa",
                "type": "any",
                "attributes": [],
            },
        ],
    }

    project = client.projects.create_from_dataset(spec=proj_spec)

    # Now we can create a task using a task repository method.
    image_list = (
        "/ppcx_domains/data/gt/quiver_images/quiver_1858_2024-07-11-08-00_2024-07-14-08-00_3days.png",
        "/ppcx_domains/data/gt/quiver_images/quiver_1829_2024-06-28-08-00_2024-07-02-08-00_4days.png",
    )
    task_spec = {
        "name": "test",
        "project_id": project.id,
    }
    task = client.tasks.create_from_data(
        spec=task_spec,
        resource_type=ResourceType.SHARE,
        resources=image_list,
    )
