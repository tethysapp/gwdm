********************************************
Setup: Web Application
********************************************

*Groundwater Data Mapper*

Prerequisites
--------------

-  Tethys Platform 3.0 with GeoServer, Postgis, and Thredds Docker containers: See:
   http://docs.tethysplatform.org

Installation
--------------
The installation is process is different for local development vs deployment. This section will walk you through both the methods.

Installation for App Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Install Tethys Platform**

Run these commands sequentially on the command line interface. Note: These instructions are for most linux systems. Most of the concepts are the same for mac/centos, except the docker installation.
You will have to install docker through the Docker documentation. Then use the tethys docker init commands to initialize the docker containers.

::

    mkdir thredds
    wget https://raw.githubusercontent.com/tethysplatform/tethys/release/scripts/install_tethys.sh
    bash install_tethys.sh --install-docker --docker-options '"-d -c thredds postgis geoserver"'


- Postgis Options
    * Enter the default options. *postgres* is the super user.

- GeoServer Options
    * Enter the options as you see fit. *admin* is the username, *geoserver* is the password for the geoserver portal.

- Thredds Options
    * Enter the options as you see fit. Be sure to link the thredds public directory to the thredds directory that was just created.

**Optional Tools to help with development**

*  pgAdmin4 - Application to view/manage postgresql database.
*  portainer - Open Source UI to manage Docker. View/manage docker images and containers.

**Install Groundwater Data Mapper App into the Portal**

Activate Tethys environment. If you followed developer installation simply type *t* and press enter. That will activate the Tethys environment.
If you installed Tethys as a conda-package activate the conda environment with the Tethys package.

You can use the following command to set alias via bashrc or you can use the command to activate the tethys-dev environment
::

    alias t='source /home/dev/miniconda/etc/profile.d/conda.sh; conda activate tethys-dev'

Clone the app repository into a directory of choice and install it
::

    git clone https://github.com/BYU-Hydroinformatics/gwdm.git
    cd gwdm
    tethys install -d

**Web Admin Setup**

Go to Web Admin Setup through the Django Site Admin Panel after logging in as admin. Username *admin* Password *pass*. If the app hasn't been configured you can simply click on the app to go the app admin panel.

Home › Tethys Apps › Installed Apps › Groundwater Data Mapper

.. figure:: /images/admin_panel.PNG
    :width: 100%
    :align: center
    :alt: Admin Panel
    :figclass: align-center

-   Set gw_thredds_directory

    *   Create a directory called groundwater under the testdata folder in thredds

        *   /home/dev/Thredds/public/testdata/groundwater

    * Paste the path to directory in the custom_settings input text box

-   Set persistent_store_setting

    * Create a persistent store service by clicking on the plus sign next to the persistent store service dropdown. Set the appropriate values based on the values that were set while initializing the postgis docker container. The following is an example, your values might vary.

        *   Engine: PostgreSQL
        *   Host: localhost
        *   Port: 5435
        *   Username: postgres
        *   Password: pass

-   Set primary geoserver setting

    * Create a spatial dataset service by clicking on the plus sign next to the spatial store service dropdown. Set the appropriate values based on the values that were set while initializing the geoserver docker container. The following is an example, your values might vary. You can skip Public Endpoint and Api Key if they haven't been configured.

        *   Engine: GeoServer
        *   Endpoint: http://127.0.0.1:8181/geoserver
        *   Username: admin
        *   Password: geoserver

-   Set primary thredds setting

    * Create a spatial dataset service by clicking on the plus sign next to the spatial store service dropdown. Set the appropriate values based on the values that were set while initializing the thredds docker container. The following is an example, your values might vary. You can skip Public Endpoint and Api Key if they haven't been configured.

        *   Engine: THREDDS
        *   Endpoint: http://127.0.0.1:8383/thredds/
        *   Username: admin
        *   Password: pass
