from tethys_sdk.app_settings import (
    PersistentStoreDatabaseSetting,
    SpatialDatasetServiceSetting,
    CustomSetting,
)
from tethys_sdk.base import TethysAppBase


class Gwdm(TethysAppBase):
    """
    Tethys app class for Ground Water Level Mapper.
    """

    name = "Groundwater Data Mapper"
    index = "home"
    icon = "gwdm/images/gw_logo.png"
    package = "gwdm"
    root_url = "gwdm"
    color = "#2c3e50"
    description = ""
    tags = ""
    enable_feedback = False
    feedback_emails = []

    controller_modules = ['controllers', 'controllers_ajax', ]

    def custom_settings(self):

        custom_settings = (
            CustomSetting(
                name="gw_data_directory",
                type=CustomSetting.TYPE_STRING,
                description="Path to the Ground Water Data Directory",
                required=True,
            ),
            CustomSetting(
                name="gw_thredds_directory",
                type=CustomSetting.TYPE_STRING,
                description="Path to the Ground Water Thredds Directory",
                required=True,
            ),
            CustomSetting(
                name="gw_thredds_catalog",
                type=CustomSetting.TYPE_STRING,
                description="Path to the Ground Water Thredds Catalog XML URL",
                required=True,
            ),
        )

        return custom_settings

    def persistent_store_settings(self):
        """
        Define Persistent Store Settings.
        """
        ps_settings = (
            PersistentStoreDatabaseSetting(
                name="gwdb",
                description="Ground Water Database",
                initializer="gwdm.model.init_db",
                required=True,
                spatial=True,
            ),
        )

        return ps_settings

    def spatial_dataset_service_settings(self):
        """
        Example spatial_dataset_service_settings method.
        """
        sds_settings = (
            SpatialDatasetServiceSetting(
                name="primary_geoserver",
                description="Geoserver for app to use",
                engine=SpatialDatasetServiceSetting.GEOSERVER,
                required=True,
            ),
            SpatialDatasetServiceSetting(
                name="primary_thredds",
                description="Thredds  for app to use",
                engine=SpatialDatasetServiceSetting.THREDDS,
                required=True,
            ),
        )

        return sds_settings
