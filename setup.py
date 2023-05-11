from setuptools import setup, find_namespace_packages
from setup_helper import find_all_resource_files

# -- Apps Definition -- #
namespace = 'tethysapp'
app_package = "gwdm"
release_package = "tethysapp-" + app_package

# -- Python Dependencies -- #
dependencies = []

# -- Get Resource File -- #
resource_files = find_all_resource_files(app_package, namespace)


setup(
    name=release_package,
    version="0.0.2",
    description="Ground Water Data Mapper",
    long_description="Ground Water Data Mapper",
    keywords="replace_keywords",
    author="Sarva Pulla",
    author_email="Sarva Pulla_email",
    url="https://github.com/BYU-Hydroinformatics/gwdm",
    license="MIT",
    packages=find_namespace_packages(),
    package_data={"": resource_files},
    include_package_data=True,
    zip_safe=False,
    install_requires=dependencies,
)
