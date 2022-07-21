/*****************************************************************************
 * FILE:    Upload Rasters
 * DATE:    25 AUGUST 2020
 * AUTHOR: Sarva Pulla
 * LICENSE: BSD 2-Clause
 *****************************************************************************/

/*****************************************************************************
 *                      LIBRARY WRAPPER
 *****************************************************************************/

var LIBRARY_OBJECT = (function() {
    // Wrap the library in a package function
    "use strict"; // And enable strict mode for this library

    /************************************************************************
     *                      MODULE LEVEL / GLOBAL VARIABLES
     *************************************************************************/
    var $rasterInput,
        $rasterModal,
        public_interface;

    /************************************************************************
     *                    PRIVATE FUNCTION DECLARATIONS
     *************************************************************************/

    var get_raster_attrs,
        upload_rasters,
        init_all,
        init_dropdown,
        init_events,
        init_jquery_vars,
        reset_dropdown,
        reset_form;

    /************************************************************************
     *                    PRIVATE FUNCTION IMPLEMENTATIONS
     *************************************************************************/
    //Reset the form when the request is made succesfully
    reset_form = function(result){
        if("success" in result){
            addSuccessMessage('Raster Successfully Uploaded!');
            $("#aquifer-select").empty().trigger('change');
            $("#variable-select").empty().trigger('change');        }
    };

    init_jquery_vars = function(){
        $rasterInput = $("#shp-upload-input");
        $rasterModal = $("#wells-modal");
    };

    init_dropdown = function () {
        $(".lat_attributes").select2({dropdownParent: $rasterModal});
        $(".lon_attributes").select2({dropdownParent: $rasterModal});
        $(".time_attributes").select2({dropdownParent: $rasterModal});
        $(".display_attributes").select2({dropdownParent: $rasterModal});
    };

    reset_dropdown = function(){
        $("#lat_attributes").html('');
        $("#lon_attributes").html('');
        $("#time_attributes").html('');
        $("#display_attributes").html('');
        $("#output-file-input").html('');
        $("#lat_attributes").val(null).trigger('change.select2');
        $("#lon_attributes").val(null).trigger('change.select2');
        $("#display_attributes").val(null).trigger('change.select2');
        $("#time_attributes").val(null).trigger('change.select2');
    };

    get_raster_attrs = function(){
        var rasters = $("#shp-upload-input")[0].files;
        if($rasterInput.val() === ""){
            addErrorMessage("Raster File cannot be empty!");
            return false;
        }else{
            reset_alert();
        }

        addInfoMessage("Getting attributes. Please wait...","message");
        var data = new FormData();
        for(var i=0;i < rasters.length;i++){
            data.append("raster", rasters[i]);
        }
        var submit_button = $("#submit-get-attributes");
        var submit_button_html = submit_button.html();
        submit_button.text('Submitting ...');
        var xhr = ajax_update_database_with_file("get-attributes", data); //Submitting the data through the ajax function, see main.js for the helper function.
        xhr.done(function(return_data){ //Reset the form once the data is added successfully
            if("success" in return_data){
                submit_button.html(submit_button_html);
                $(".attributes").removeClass('d-none');
                reset_dropdown();
                var attributes = return_data["attributes"];
                if("error" in attributes){
                    addErrorMessage(attributes['error']);
                    submit_button.html(submit_button_html);
                }else{
                    $rasterModal.modal('show');
                    var empty_opt = '<option value="" selected disabled>Select item...</option>';
                    $("#lat_attributes").append(empty_opt);
                    $("#lon_attributes").append(empty_opt);
                    $("#time_attributes").append(empty_opt);
                    $("#display_attributes").append(empty_opt);
                    // $("#meta_attributes").append(empty_opt);
                    attributes["coords"].forEach(function(attr,i){
                        var lat_option = new Option(attr, attr);
                        var lon_option = new Option(attr, attr);
                        var time_option = new Option(attr, attr)
                        $("#lat_attributes").append(lat_option);
                        $("#lon_attributes").append(lon_option);
                        $("#time_attributes").append(time_option);
                    });
                    attributes["keys"].forEach(function(attr,i){
                        var display_option = new Option(attr, attr);
                        $("#display_attributes").append(display_option);
                    });
                    $(".add").removeClass('d-none');
                    addSuccessMessage('Got Attributes Successfully!');
                }
            }else{
                addErrorMessage(return_data['error']);
                submit_button.html(submit_button_html);
                reset_dropdown();
            }
        });
    };

    $("#submit-get-attributes").click(get_raster_attrs);

    upload_rasters = function(){
        reset_alert();
        var region = $("#region-select option:selected").val();
        var aquifer = $("#aquifer-select option:selected").text();
        var variable = $("#variable-select option:selected").val();
        var file_name = $("#output-file-input").val();
        var clip = $("#clip-select option:selected").val();
        var ncfiles = $("#shp-upload-input")[0].files;
        var lat = $("#lat_attributes option:selected").val();
        var lon = $("#lon_attributes option:selected").val();
        var time = $("#time_attributes option:selected").val();
        var display = $("#display_attributes option:selected").val();
        if(aquifer === ""){
            addErrorMessage("Aquifer cannot be empty! Please select an Aquifer.");
            return false;
        }else{
            reset_alert();
        }
        if(variable === ""){
            addErrorMessage("Variable cannot be empty! Please select a Variable.");
            return false;
        }else{
            reset_alert();
        }
        if(file_name === ""){
            addErrorMessage("Output File Name cannot be empty! Please enter a value.");
            return false;
        }else{
            reset_alert();
        }
        if(lat === ""){
            addErrorMessage("Lat cannot be empty! Please select a Lat variable.");
            return false;
        }else{
            reset_alert();
        }
        if(lon === ""){
            addErrorMessage("Lon cannot be empty! Please select a Lon variable.");
            return false;
        }else{
            reset_alert();
        }
        if(time === ""){
            addErrorMessage("Time cannot be empty! Please select a time variable.");
            return false;
        }else{
            reset_alert();
        }
        if(display === ""){
            addErrorMessage("Display cannot be empty! Please select a display Variable.");
            return false;
        }else{
            reset_alert();
        }
        addInfoMessage("Uploading Rasters. Please wait...","message");

        var data = new FormData();
        data.append("region", region);
        data.append("aquifer", aquifer);
        data.append("variable", variable);
        data.append("file_name", file_name);
        data.append("clip", clip);
        data.append("lat", lat);
        data.append("lon", lon);
        data.append("time_var", time);
        data.append("display_var", display);

        for(var i=0;i < ncfiles.length;i++){
            data.append("ncfiles",ncfiles[i]);
        }
        var submit_button = $("#submit-add-rasters");
        var submit_button_html = submit_button.html();
        submit_button.text('Uploading Rasters ...');
        var xhr = ajax_update_database_with_file("submit", data); //Submitting the data through the ajax function, see main.js for the helper function.
        xhr.done(function(return_data){ //Reset the form once the data is added successfully
            if("success" in return_data){
                submit_button.html(submit_button_html);
                reset_form(return_data);
            }else{
                submit_button.html(submit_button_html);
                addErrorMessage(return_data['error']);
                console.log(return_data['error'])
            }
        });

    };

    $(".submit-add-rasters").click(upload_rasters);

    init_all = function(){
        init_jquery_vars();
        init_dropdown();
    };

    /************************************************************************
     *                        DEFINE PUBLIC INTERFACE
     *************************************************************************/
    /*
     * Library object that contains public facing functions of the package.
     * This is the object that is returned by the library wrapper function.
     * See below.
     * NOTE: The functions in the public interface have access to the private
     * functions of the library because of JavaScript function scope.
     */
    public_interface = {

    };

    /************************************************************************
     *                  INITIALIZATION / CONSTRUCTOR
     *************************************************************************/

    // Initialization: jQuery function that gets called when
    // the DOM tree finishes loading
    $(function() {
        init_all();
        $('#variable-select').select2('val', '');

        $("#region-select").change(function(){
            var region = $("#region-select option:selected").val();
            var xhr = ajax_update_database("get-aquifers", {'id': region}); //Submitting the data through the ajax function, see main.js for the helper function.
            xhr.done(function(return_data){ //Reset the form once the data is added successfully
                if("success" in return_data){
                    var options = return_data["aquifers_list"];
                    var var_options = return_data["variables_list"];
                    $("#aquifer-select").html('');
                    // $("#variable-select").html('');
                    // $("#variable-select").select2({'multiple': true,  placeholder: "Select a Variable(s)"});
                    // $("#aquifer-select").select2({'multiple': false,  placeholder: "Select an Aquifer(s)"});
                    var empty_opt = '<option value="" selected disabled>Select item...</option>';
                    // var var_empty_opt = '<option value="" selected disabled>Select item...</option>';
                    $("#aquifer-select").append(empty_opt);
                    // $("#variable-select").append(var_empty_opt);
                    options.forEach(function(attr,i){
                        var aquifer_option = new Option(attr[0], attr[1]);
                        $("#aquifer-select").append(aquifer_option);
                    });
                    // var_options.forEach(function(attr, i){
                    //     var var_option = new Option(attr[0], attr[1]);
                    //     $("#variable-select").append(var_option);
                    // });
                }else{
                    addErrorMessage(return_data['error']);
                }
            });
        }).change();

    });

    return public_interface;

}()); // End of package wrapper
// NOTE: that the call operator (open-closed parenthesis) is used to invoke the library wrapper
// function immediately after being parsed.