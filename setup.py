from setuptools import setup, find_namespace_packages
from tethys_apps.app_installation import find_resource_files

# -- Apps Definition -- #
app_package = "gwdm"
release_package = "tethysapp-" + app_package

# -- Python Dependencies -- #
dependencies = []

# -- Get Resource File -- #
resource_files = find_resource_files(
    "tethysapp/" + app_package + "/templates", "tethysapp/" + app_package
)
resource_files += find_resource_files(
    "tethysapp/" + app_package + "/public", "tethysapp/" + app_package
)
resource_files += find_resource_files(
    "tethysapp/" + app_package + "/workspaces", "tethysapp/" + app_package
)

setup(
    name=release_package,
    version="0.0.2",
    tethys_version=['3', '4'],
    description="Ground Water Data Mapper",
    long_description="",
    keywords="",
    author="Sarva Pulla",
    author_email="",
    url="https://github.com/BYU-Hydroinformatics/gwdm",
    license="MIT",
    packages=find_namespace_packages(),
    package_data={"": resource_files},
    include_package_data=True,
    zip_safe=False,
    install_requires=dependencies,
)
