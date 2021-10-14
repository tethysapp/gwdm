from django.contrib import messages
from django.contrib.auth.decorators import user_passes_test
from django.http.response import HttpResponseRedirect
from django.shortcuts import render, reverse, redirect
from tethys_sdk.compute import get_scheduler
from tethys_sdk.gizmos import Button, TextInput, SelectInput
from tethys_sdk.gizmos import JobsTable, PlotlyView

from .app import Gwdm as app
from .model import Variable
from .utils import (
    get_regions,
    get_aquifers_list,
    get_geoserver_status,
    get_thredds_status,
    get_num_wells,
    get_num_measurements,
    get_num_rasters,
    get_variable_list,
    get_metrics,
    geoserver_text_gizmo,
    get_region_select,
    get_region_variable_select,
    get_aquifer_select,
    get_variable_select,
    thredds_text_gizmo,
    get_region_name,
    get_session_obj,
)
from .utils import user_permission_test

# get job manager for the app
job_manager = app.get_job_manager()


def home(request):
    """
    Controller for the app home page.
    """
    region_select = get_region_select()

    context = {"region_select": region_select}

    return render(request, "gwdm/home.html", context)


def config(request):
    """
    Controller for helping setup the app
    """
    add_geoserver_config = Button(
        display_text="Configure GeoServer",
        icon="glyphicon glyphicon-plus",
        style="success",
        name="submit",
        attributes={"id": "submit"},
        classes="add",
    )
    geoserver_status = get_geoserver_status()
    thredds_status = get_thredds_status()
    context = {
        "add_geoserver_config": add_geoserver_config,
        "workspace_status": geoserver_status["workspace_status"],
        "store_status": geoserver_status["store_status"],
        "layer_status": geoserver_status["layer_status"],
        "directory_status": thredds_status["directory_status"],
    }
    return render(request, "gwdm/config.html", context)


def metrics(request):
    """
    Controller for the app metrics page.
    """
    regions_list = get_regions()
    num_regions = len(regions_list)
    variables_list = get_variable_list()
    num_variables = len(variables_list)
    aquifers_list = get_aquifers_list()
    num_aquifers = len(aquifers_list)
    num_wells = get_num_wells()
    num_measurements = get_num_measurements()
    num_rasters = get_num_rasters()
    metrics_plot = PlotlyView(get_metrics(), show_link=True)

    context = {
        "num_regions": num_regions,
        "num_variables": num_variables,
        "num_aquifers": num_aquifers,
        "num_wells": num_wells,
        "num_measurements": num_measurements,
        "num_rasters": num_rasters,
        "metrics_plot": metrics_plot,
    }

    return render(request, "gwdm/metrics.html", context)


def region_map(request):
    """
    Controller for the Region Map home page.
    """
    info = request.GET

    region_id = info.get("region-select")
    region_select = get_region_select()
    region_name = get_region_name(int(region_id))
    aquifer_select = get_aquifer_select(int(region_id), aquifer_id=True)
    geoserver_text_input = geoserver_text_gizmo()
    thredds_text_input = thredds_text_gizmo()
    variable_select = get_region_variable_select(region_id=region_id)
    # variable_select = get_variable_select()
    select_interpolation = SelectInput(
        display_text="Interpolation Layer",
        name="select-interpolation",
        options=[],
        multiple=False,
    )

    region_text_input = TextInput(
        display_text="Region",
        name="region-text-input",
        placeholder=region_id,
        attributes={"value": region_id, "region-name": region_name},
        classes="hidden",
    )

    context = {
        "region_select": region_select,
        "aquifer_select": aquifer_select,
        "variable_select": variable_select,
        "geoserver_text_input": geoserver_text_input,
        "thredds_text_input": thredds_text_input,
        "select_interpolation": select_interpolation,
        "region_text_input": region_text_input,
        "region_name": region_name,
    }

    return render(request, "gwdm/region_map.html", context)


@user_passes_test(user_permission_test)
def interpolation(request):
    region_select = get_region_select()
    aquifer_select = get_aquifer_select(None)
    variable_select = get_region_variable_select(None)
    output_file_input = TextInput(
        display_text="Output Filename. Unique Identifier to differentiate from different interpolation runs.",
        name="output-file-input"
    )
    raster_extent_select = SelectInput(
        display_text="Raster Extent",
        name="select-raster-extent",
        multiple=False,
        options=[
            ("Aquifer Extent", "aquifer"),
            ("Well Data Extent", "wells")
        ]
    )
    temporal_interpolation = SelectInput(
        display_text="Temporal Interpolation Method",
        name="select-temporal-interpolation",
        multiple=False,
        options=[
            # ("Pchip Interpolation", "pchip"),
            ("Multi-Linear Regression", "MLR")
        ],
        initial="Multi-Linear Regression",
    )
    raster_interval = SelectInput(
        display_text="Select Time Interval",
        name="select-raster-interval",
        multiple=False,
        options=[
            ("1 year", 12),
            ("2 years", 24),
            ("3 years", 36),
            ("4 years", 48),
            ("5 years", 60),
        ]
    )
    min_samples = SelectInput(
        display_text="Minimum Water Level Samples per Well",
        name="min-samples",
        options=[
            ("1 Sample", 1),
            ("2 Samples", 2),
            ("5 Samples", 5),
            ("10 Samples", 10),
            ("15 Samples", 15),
            ("20 Samples", 20),
            ("25 Samples", 25),
            ("50 Samples", 50),
        ],
        initial="10 Samples",
    )

    gap_size = TextInput(
        display_text="Enter GAP Size",
        name="gap-size-input",
        initial="365 days",
    )

    pad = TextInput(
        display_text="Enter Pad Value",
        name="pad-input",
        initial="90",
    )

    spacing = TextInput(
        display_text="Enter Spacing",
        name="spacing-input",
        initial="1MS",
    )
    # 'gap_size': '365 days',
    # 'pad': '90',
    # 'spacing': '1MS'
    add_button = Button(
        display_text="Submit",
        icon="glyphicon glyphicon-plus",
        style="primary",
        name="submit",
        attributes={"id": "submit"},
        # href=reverse('gwdm:run-dask', kwargs={'job_type': 'delayed'}),
        classes="add",
    )

    context = {
        "region_select": region_select,
        "aquifer_select": aquifer_select,
        "variable_select": variable_select,
        "output_file_input": output_file_input,
        "raster_extent_select": raster_extent_select,
        "temporal_interpolation": temporal_interpolation,
        "raster_interval": raster_interval,
        "min_samples": min_samples,
        "gap_size": gap_size,
        "pad": pad,
        "spacing": spacing,
        "add_button": add_button,
    }
    return render(request, "gwdm/interpolation.html", context)


@user_passes_test(user_permission_test)
def add_region(request):
    """
    Controller for add region
    """

    region_text_input = TextInput(
        display_text="Region Name",
        name="region-text-input",
        placeholder="e.g.: West Africa",
        attributes={"id": "region-text-input"},
    )

    add_button = Button(
        display_text="Add Region",
        icon="glyphicon glyphicon-plus",
        style="primary",
        name="submit-add-region",
        attributes={"id": "submit-add-region"},
        classes="add",
    )

    context = {"region_text_input": region_text_input, "add_button": add_button}

    return render(request, "gwdm/add_region.html", context)


@user_passes_test(user_permission_test)
def update_region(request):
    id_input = TextInput(
        display_text="Region ID",
        name="id-input",
        placeholder="",
        attributes={"id": "id-input", "readonly": "true"},
    )

    region_text_input = TextInput(
        display_text="Region Name",
        name="region-text-input",
        placeholder="e.g.: West Africa",
        attributes={"id": "region-text-input"},
    )

    geoserver_text_input = geoserver_text_gizmo()

    context = {
        "id_input": id_input,
        "region_text_input": region_text_input,
        "geoserver_text_input": geoserver_text_input,
    }
    return render(request, "gwdm/update_region.html", context)


@user_passes_test(user_permission_test)
def add_aquifer(request):
    """
    Controller for add aquifer
    """
    region_select = get_region_select()
    aquifer_text_input = TextInput(
        display_text="Aquifer Name",
        name="aquifer-text-input",
        placeholder="e.g.: Niger",
        attributes={"id": "aquifer-text-input"},
    )

    add_button = Button(
        display_text="Add Aquifer",
        icon="glyphicon glyphicon-plus",
        style="primary",
        name="submit-add-aquifer",
        attributes={"id": "submit-add-aquifer"},
        classes="add hidden",
    )

    attributes_button = Button(
        display_text="Get Attributes",
        icon="glyphicon glyphicon-plus",
        style="primary",
        name="submit-get-attributes",
        attributes={"id": "submit-get-attributes"},
    )

    context = {
        "aquifer_text_input": aquifer_text_input,
        "region_select": region_select,
        "attributes_button": attributes_button,
        "add_button": add_button,
    }

    return render(request, "gwdm/add_aquifer.html", context)


@user_passes_test(user_permission_test)
def update_aquifer(request):
    id_input = TextInput(
        display_text="Aquifer ID",
        name="id-input",
        placeholder="",
        attributes={"id": "id-input", "readonly": "true"},
    )

    aquifer_text_input = TextInput(
        display_text="Aquifer Name",
        name="aquifer-text-input",
        placeholder="e.g.: Abod",
        attributes={"id": "aquifer-text-input"},
    )

    aquifer_id_input = TextInput(
        display_text="Aquifer ID",
        name="aquifer-id-input",
        placeholder="e.g.: 23",
        attributes={"id": "aquifer-id-input"},
    )

    geoserver_text_input = geoserver_text_gizmo()

    context = {
        "id_input": id_input,
        "geoserver_text_input": geoserver_text_input,
        "aquifer_text_input": aquifer_text_input,
        "aquifer_id_input": aquifer_id_input,
    }
    return render(request, "gwdm/update_aquifer.html", context)


@user_passes_test(user_permission_test)
def add_wells(request):
    """
    Controller for add wells
    """

    region_select = get_region_select()

    aquifer_select = get_aquifer_select(None)

    attributes_button = Button(
        display_text="Get Attributes",
        icon="glyphicon glyphicon-plus",
        style="primary",
        name="submit-get-attributes",
        attributes={"id": "submit-get-attributes"},
    )

    context = {
        "aquifer_select": aquifer_select,
        "region_select": region_select,
        "attributes_button": attributes_button,
    }

    return render(request, "gwdm/add_wells.html", context)


@user_passes_test(user_permission_test)
def edit_wells(request):
    geoserver_text_input = geoserver_text_gizmo()
    region_select = get_region_select()
    aquifer_select = get_aquifer_select(None)

    context = {
        "geoserver_text_input": geoserver_text_input,
        "region_select": region_select,
        "aquifer_select": aquifer_select,
    }
    return render(request, "gwdm/edit_wells.html", context)


@user_passes_test(user_permission_test)
def delete_wells(request):
    region_select = get_region_select()
    aquifer_select = get_aquifer_select(None)
    delete_button = Button(
        display_text="Delete Wells",
        icon="glyphicon glyphicon-minus",
        style="danger",
        name="submit-delete-wells",
        attributes={"id": "submit-delete-wells"},
        classes="delete",
    )

    context = {
        "region_select": region_select,
        "aquifer_select": aquifer_select,
        "delete_button": delete_button,
    }
    return render(request, "gwdm/delete_wells.html", context)


@user_passes_test(user_permission_test)
def upload_rasters(request):
    region_select = get_region_select()
    aquifer_select = get_aquifer_select(None)
    variable_select = get_variable_select()
    add_button = Button(
        display_text="Add Rasters",
        icon="glyphicon glyphicon-plus",
        style="success",
        name="submit-upload-rasters",
        attributes={"id": "submit-upload-rasters"},
        classes="delete",
    )

    context = {
        "region_select": region_select,
        "aquifer_select": aquifer_select,
        "variable_select": variable_select,
        "add_button": add_button,
    }
    return render(request, "gwdm/upload_rasters.html", context)


@user_passes_test(user_permission_test)
def delete_rasters(request):
    region_select = get_region_select()
    aquifer_select = get_aquifer_select(None)
    variable_select = get_variable_select()
    select_interpolation = SelectInput(
        display_text="Interpolation Layer",
        name="select-interpolation",
        options=[],
        multiple=False,
    )
    delete_button = Button(
        display_text="Delete Rasters",
        icon="glyphicon glyphicon-minus",
        style="danger",
        name="submit-delete-rasters",
        attributes={"id": "submit-delete-rasters"},
        classes="delete",
    )

    context = {
        "region_select": region_select,
        "aquifer_select": aquifer_select,
        "variable_select": variable_select,
        "select_interpolation": select_interpolation,
        "delete_button": delete_button,
    }
    return render(request, "gwdm/delete_rasters.html", context)


@user_passes_test(user_permission_test)
def add_measurements(request):
    region_select = get_region_select()

    variable_select = get_variable_select()

    aquifer_select = get_aquifer_select(None)

    attributes_button = Button(
        display_text="Get Attributes",
        icon="glyphicon glyphicon-plus",
        style="primary",
        name="submit-get-attributes",
        attributes={"id": "submit-get-attributes"},
    )
    format_text_input = TextInput(
        display_text="Date Format",
        name="format-text-input",
        placeholder="e.g.: mm-dd-yyyy",
        attributes={"id": "format-text-input"},
    )

    context = {
        "region_select": region_select,
        "aquifer_select": aquifer_select,
        "variable_select": variable_select,
        "attributes_button": attributes_button,
        "format_text_input": format_text_input,
    }
    return render(request, "gwdm/add_measurements.html", context)


@user_passes_test(user_permission_test)
def update_measurements(request):
    region_select = get_region_select()
    aquifer_select = get_aquifer_select(None)
    variable_select = get_region_variable_select(None)
    delete_button = Button(
        display_text="Delete Measurements",
        icon="glyphicon glyphicon-minus",
        style="danger",
        name="submit-delete-measurements",
        attributes={"id": "submit-delete-measurements"},
        classes="delete",
    )

    context = {
        "region_select": region_select,
        "aquifer_select": aquifer_select,
        "variable_select": variable_select,
        "delete_button": delete_button,
    }
    return render(request, "gwdm/update_measurements.html", context)


@user_passes_test(user_permission_test)
def add_variable(request):
    name_error = ""
    units_error = ""
    desc_error = ""

    if request.method == "POST" and "submit-add-variable" in request.POST:
        has_errors = False

        name = request.POST.get("name", None)
        units = request.POST.get("units", None)
        desc = request.POST.get("desc", None)

        if not name:
            has_errors = True
            name_error = "Name is required."
        if not units:
            has_errors = True
            units_error = "Units are required"
        if not desc:
            has_errors = True
            desc_error = "Description is required"

        if not has_errors:
            session = get_session_obj()
            var_obj = Variable(name=name, units=units, description=desc)
            session.add(var_obj)
            session.commit()
            session.close()
            return redirect(reverse("gwdm:home"))

        messages.error(request, "Please fix errors.")

    name_text_input = TextInput(
        display_text="Variable Name",
        name="name",
        placeholder="e.g.: Ground Water Depth",
        attributes={"id": "variable-text-input"},
        error=name_error,
    )

    units_text_input = TextInput(
        display_text="Variable Units",
        name="units",
        placeholder="e.g.: m",
        attributes={"id": "units-text-input"},
        error=units_error,
    )

    desc_text_input = TextInput(
        display_text="Variable Description",
        name="desc",
        placeholder="e.g.: m",
        attributes={"id": "desc-text-input"},
        error=desc_error,
    )

    add_button = Button(
        display_text="Add Variable",
        icon="glyphicon glyphicon-plus",
        style="primary",
        name="submit-add-variable",
        attributes={"form": "add-variable-form"},
        submit=True,
        classes="add",
    )

    context = {
        "name_text_input": name_text_input,
        "desc_text_input": desc_text_input,
        "units_text_input": units_text_input,
        "add_button": add_button,
    }

    return render(request, "gwdm/add_variable.html", context)


@user_passes_test(user_permission_test)
def update_variable(request):
    id_input = TextInput(
        display_text="Variable ID",
        name="id-input",
        placeholder="",
        attributes={"id": "id-input", "readonly": "true"},
    )

    name_text_input = TextInput(
        display_text="Variable Name",
        name="variable-text-input",
        placeholder="e.g.: Ground Water Depth",
        attributes={"id": "variable-text-input"},
    )

    units_text_input = TextInput(
        display_text="Variable Units",
        name="units-text-input",
        placeholder="e.g.: m",
        attributes={"id": "units-text-input"},
    )

    desc_text_input = TextInput(
        display_text="Variable Description",
        name="desc-text-input",
        placeholder="e.g.: m",
        attributes={"id": "desc-text-input"},
    )

    context = {
        "id_input": id_input,
        "name_text_input": name_text_input,
        "units_text_input": units_text_input,
        "desc_text_input": desc_text_input,
    }
    return render(request, "gwdm/update_variable.html", context)


def run_job(request, job_type):
    """
    Controller for the app home page.
    """
    # Get test_scheduler app. This scheduler needs to be in the database.
    scheduler = get_scheduler(name="dask_local")

    if job_type.lower() == "delayed":
        from .job_functions import delayed_job

        # Create dask delayed object
        delayed = delayed_job()
        dask = job_manager.create_job(
            job_type="DASK",
            name="dask_delayed",
            user=request.user,
            scheduler=scheduler,
        )

        # Execute future
        dask.execute(delayed)

    return HttpResponseRedirect(reverse("gwdm:jobs-table"))


def jobs_table(request):
    # Use job manager to get all the jobs.
    jobs = job_manager.list_jobs(order_by="-id", filters=None)

    # Table View
    jobs_table_options = JobsTable(
        jobs=jobs,
        column_fields=("id", "name", "description", "creation_time"),
        hover=True,
        striped=False,
        bordered=False,
        condensed=False,
        results_url="gwdm:result",
        refresh_interval=1000,
        delete_btn=True,
        show_detailed_status=True,
    )

    home_button = Button(
        display_text="Home",
        name="home_button",
        attributes={"data-toggle": "tooltip", "data-placement": "top", "title": "Home"},
        href=reverse("gwdm:home"),
    )

    context = {"jobs_table": jobs_table_options, "home_button": home_button}

    return render(request, "gwdm/jobs_table.html", context)


def result(request, job_id):
    # Use job manager to get the given job.
    job = job_manager.get_job(job_id=job_id)

    # Get result and name
    job_result = job.result
    name = job.name

    home_button = Button(
        display_text="Home",
        name="home_button",
        attributes={"data-toggle": "tooltip", "data-placement": "top", "title": "Home"},
        href=reverse("gwdm:home"),
    )

    jobs_button = Button(
        display_text="Show All Jobs",
        name="dask_button",
        attributes={
            "data-toggle": "tooltip",
            "data-placement": "top",
            "title": "Show All Jobs",
        },
        href=reverse("gwdm:jobs-table"),
    )

    context = {
        "result": job_result,
        "name": name,
        "home_button": home_button,
        "jobs_button": jobs_button,
    }

    return render(request, "gwdm/results.html", context)


def error_message(request):
    messages.add_message(request, messages.ERROR, "Invalid Scheduler!")
    return redirect(reverse("gwdm:home"))
