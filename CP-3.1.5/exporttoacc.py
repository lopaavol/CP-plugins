# coding=utf-8

"""
ExportToACC
===========

**ExportToACC** exports measurements into one or more files that can be
opened in Advanced Cell Classifier. The module exports file named "featureNames.acc" that includes the names and order of the exported features. The features are exported in separate files for each field/well and includes single cell in each line having features separated with single space (in the order given by featureNames.acc).

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For details on the nomenclature used by CellProfiler for the exported
measurements, see *Help > General Help > How Measurements Are Named*
"""

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2015 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

# Modified from exporttospreadsheet.py by Lassi Paavolainen 2015
# Modified from CellProfiler 2.1 version to CellProfiler 3.x 2018

import logging
logger = logging.getLogger(__name__)
import base64
import csv
import errno
import numpy as np
import os
import sys

import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
from cellprofiler.measurement import IMAGE, EXPERIMENT
from cellprofiler.preferences import get_absolute_path, get_output_file_name
from cellprofiler.preferences import ABSPATH_OUTPUT, ABSPATH_IMAGE, get_headless
from cellprofiler.modules._help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF, IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT
from cellprofiler.preferences import \
     standardize_default_folder_names, DEFAULT_INPUT_FOLDER_NAME, \
     DEFAULT_OUTPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
     DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME

DELIMITER = " "

"""The caption for the image set number"""
IMAGE_NUMBER = "ImageNumber"

"""The caption for the object # within an image set"""
OBJECT_NUMBER = "ObjectNumber"

DIR_CUSTOM = "Custom folder"
DIR_CUSTOM_WITH_METADATA = "Custom folder with metadata"

NANS_AS_NULLS = "Null"
NANS_AS_NANS = "NaN"

"""The file name of acc feature list file"""
ACC_FILE_NAME = "featureNames.acc"

"""Remove features with following substrings"""
REMOVE_FEAT = ("Location","Number_Object_Number","_Center_")

class ExportToACC(cpm.Module):
    module_name = 'ExportToACC'
    category = ["File Processing","Data Tools"]
    variable_revision_number = 2
    
    def create_settings(self):
        self.directory = cps.DirectoryPath(
            "Output file location",
            dir_choices = [
                ABSOLUTE_FOLDER_NAME, 
                DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME,
                DEFAULT_INPUT_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME ],doc="""
            This setting lets you choose the folder for the output
            files. %(IO_FOLDER_CHOICE_HELP_TEXT)s
            
            %(IO_WITH_METADATA_HELP_TEXT)s
"""%globals())
        self.directory.dir_choice = DEFAULT_OUTPUT_FOLDER_NAME

        self.wants_file_name_suffix = cps.Binary(
            "Append a suffix to the file name?", False, doc = """
            Select *"%(YES)s"* to add a suffix to the file name.
            Select *"%(NO)s"* to use the file name as-is.
"""%globals())

        self.file_name_suffix = cps.Text(
            "Text to append to the file name",
            "", metadata = True, doc="""
            "*(Used only when constructing the filename from the image filename)*"
            Enter the text that should be appended to the filename specified above.
""")

        self.wants_overwrite_without_warning = cps.Binary(
            "Overwrite without warning?", False,
            doc="""This setting either prevents or allows overwriting of
            old .txt files by **ExportToACC** without confirmation.
            Select "*%(YES)s*" to overwrite without warning any .txt file 
            that already exists. Select "*%(NO)s*" to prompt before overwriting
            when running CellProfiler in the GUI and to fail when running
            headless.
"""%globals())

        self.nan_representation = cps.Choice(
            "Representation of Nan/Inf", [NANS_AS_NANS, NANS_AS_NULLS], doc = """
            This setting controls the output for numeric fields
            if the calculated value is infinite (*"Inf"*) or undefined (*"NaN*").
            CellProfiler will produce Inf or NaN values under certain rare
            circumstances, for instance when calculating the mean intensity
            of an object within a masked region of an image.

            - "*%(NANS_AS_NULLS)s:*" Output these values as empty fields.
            - "*%(NANS_AS_NANS)s:*" Output them as the strings "NaN", "Inf" or "-Inf".

"""%globals())
        
        self.pick_columns = cps.Binary(
            "Select the measurements to export", False, doc = """
            Select *"%(YES)s"* to provide a button that allows you to select which measurements you want to export.
            This is useful if you know exactly what measurements you want included in the final spreadheet(s). """%globals())
        
        self.columns = cps.MeasurementMultiChoice(
            "Press button to select measurements to export",doc = """
            "*(Used only when selecting the columns of measurements to export)*"
            This setting controls the columns to be exported. Press
            the button and check the measurements or categories to export.""")

        self.file_image_name = cps.FileImageNameSubscriber(
            "Select image name for file prefix",
            cps.NONE,doc="""
            Select an image loaded using **NamesAndTypes**. The original filename will be
            used as the prefix for the output filename."""%globals())
    
    def settings(self):
        """Return the settings in the order used when storing """
        result = [self.pick_columns,
                  self.directory,
                  self.columns, self.nan_representation,
                  self.wants_file_name_suffix, self.file_name_suffix,
                  self.wants_overwrite_without_warning,
                  self.file_image_name]

        return result

    def visible_settings(self):
        """Return the settings as seen by the user"""
        result = [self.directory, self.file_image_name, self.wants_file_name_suffix]
        if self.wants_file_name_suffix:
            result += [self.file_name_suffix]
        result += [self.wants_overwrite_without_warning, self.nan_representation, self.pick_columns]
        if self.pick_columns:
            result += [self.columns]

        return result

    def validate_module(self, pipeline):
        '''Test the module settings to make sure they are internally consistent'''
        '''Make sure metadata tags exist'''
        if self.wants_file_name_suffix.value:
            text_str = self.file_name_suffix.value
            undefined_tags = pipeline.get_undefined_metadata_tags(text_str)
            if len(undefined_tags) > 0:
                raise cps.ValidationError("%s is not a defined metadata tag. Check the metadata specifications in your load modules" %
                                     undefined_tags[0],
                                     self.file_name_suffix)

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError("ExportToACC will not produce output in Test Mode",
                                      self.directory)
        
    def prepare_run(self, workspace):
        '''Prepare an image set to be run
        
        workspace - workspace with image set populated (at this point)
        
        returns False if analysis can't be done
        '''
        return self.check_overwrite(workspace)
    
    def run(self, workspace):
        # all of the work is done in post_run()
        if self.show_window:
            image_set_number = workspace.measurements.image_set_number
            header = ["Filename"]
            columns = []
            path = self.make_image_file_name(workspace, image_set_number)
            columns.append((path,))
            workspace.display_data.header = header
            workspace.display_data.columns = columns

    def display(self, workspace, figure):
        figure.set_subplots((1, 1,))
        if workspace.display_data.columns is None:
            figure.subplot_table(
                0, 0, [["Data written to acc files"]])
        elif workspace.pipeline.test_mode:
            figure.subplot_table(
                0, 0, [["Data not written to acc files in test mode"]])
        else:
            figure.subplot_table(0, 0, 
                                 workspace.display_data.columns,
                                 col_labels = workspace.display_data.header)
    
    def run_as_data_tool(self, workspace):
        '''Run the module as a data tool
        
        For ExportToACC, we do the "post_run" method in order to write
        out the .txt files as if the experiment had just finished.
        '''
        #
        # Set the measurements to the end of the list to mimic the state
        # at the end of the run.
        #
        m = workspace.measurements
        m.image_set_number = m.image_set_count
        self.post_run(workspace)
        
    def post_run(self, workspace):
        '''Save measurements at end of run'''
        #
        # Don't export in test mode
        #
        #if workspace.pipeline.test_mode:
        #    return
        object_names = self.filter_object_names(workspace.measurements.get_object_names())
        self.run_objects(object_names, workspace)
        
    def should_stop_writing_measurements(self):
        '''All subsequent modules should not write measurements'''
        return True

    def get_metadata_groups(self, workspace, settings_group = None):
        '''Find the metadata groups that are relevant for creating the file name
        
        workspace - the workspace with the image set metadata elements and
                    grouping measurements populated.
        settings_group - if saving individual objects, this is the settings
                         group that controls naming the files.
        '''
        if settings_group is None or settings_group.wants_automatic_file_name:
            tags = []
        else:
            tags = cpmeas.find_metadata_tokens(settings_group.file_name.value)
        if self.directory.is_custom_choice:
            tags += cpmeas.find_metadata_tokens(self.directory.custom_path)
        metadata_groups = workspace.measurements.group_by_metadata(tags)
        return metadata_groups
    
    def run_objects(self, object_names, workspace, settings_group = None):
        """Create a file based on the object names
        
        object_names - a sequence of object names (or Image or Experiment)
                       which tell us which objects get piled into each file
        workspace - get the images from here.
        settings_group - if present, use the settings group for naming.
        
        """
        metadata_groups = self.get_metadata_groups(workspace, settings_group)
        for metadata_group in metadata_groups:
            self.make_object_file(object_names, metadata_group.image_numbers, 
                                  workspace, settings_group)
    
    def make_full_filename(self, file_name, workspace = None, image_set_number = None):
        """Convert a file name into an absolute path
        
        We do a few things here:
        * apply metadata from an image set to the file name if an 
          image set is specified
        * change the relative path into an absolute one using the "." and "&"
          convention
        * Create any directories along the path
        """
        if image_set_number is not None and workspace is not None:
            file_name = workspace.measurements.apply_metadata(file_name,
                                                              image_set_number)
        measurements = None if workspace is None else workspace.measurements
        path_name = self.directory.get_absolute_path(measurements, 
                                                     image_set_number)
        file_name = os.path.join(path_name, file_name)
        path, fname = os.path.split(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, fname)

    def extension(self):
        '''Return the appropriate extension for the txt file name
        
        The appropriate extension is "txt"
        '''
        return "txt"
    
    def make_image_file_name(
        self, workspace, image_set_number, settings_group = None):
        '''Make file name for objects measured from an image
        
        :param workspace: the current workspace
        :param image_set_number: the current image set number
        :param settings_group: the settings group used to name the file
        '''
        imagename = workspace.measurements.get_measurement(IMAGE,"FileName_"+self.file_image_name.value,image_set_number)
        filename = "%s"%os.path.splitext(imagename)[0]
        if self.wants_file_name_suffix:
            suffix = self.file_name_suffix.value
            suffix = workspace.measurements.apply_metadata(suffix, image_set_number)
            filename += suffix
        filename = "%s.%s"%(filename, self.extension())
        return self.make_full_filename(filename, workspace, image_set_number)

    def check_overwrite(self, workspace):
        """Make sure it's ok to overwrite any existing files before starting run
        
        workspace - workspace with all image sets already populated
        
        returns True if ok to proceed, False if user cancels
        """
        if self.wants_overwrite_without_warning:
            return True
        
        files_to_check = []
        metadata_groups = self.get_metadata_groups(workspace)
        for metadata_group in metadata_groups:
            image_number = metadata_group.image_numbers[0]
            files_to_check.append(self.make_image_file_name(workspace, image_number))
        
        files_to_overwrite = filter(os.path.isfile, files_to_check)
        if len(files_to_overwrite) > 0:
            if get_headless():
                logger.error("ExportToACC is configured to refrain from overwriting files and the following file(s) already exist: %s" %
                             ", ".join(files_to_overwrite))
                return False
            msg = "Overwrite the following file(s)?\n" +\
                "\n".join(files_to_overwrite)
            import wx
            result = wx.MessageBox(
                msg, caption="ExportToACC: Overwrite existing files",
                style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
            if result != wx.YES:
                return False
        
        return True
    
    def filter_columns(self, features, object_name):
        if self.pick_columns:
            columns = [
                self.columns.get_measurement_feature(x)
                for x in self.columns.selections
                if self.columns.get_measurement_object(x) == object_name]
                                
            columns = set(columns)
            features = [x for x in features if x in columns]

        return features

    def filter_object_names(self, object_names):
        object_names.remove('Image')
        object_names.remove('Experiment')
        return object_names
        
    def make_object_file(self, object_names, image_set_numbers, workspace,
                         settings_group = None):
        """Make a file containing object measurements
        
        object_names - sequence of names of the objects whose measurements
                       will be included
        image_set_numbers -  the image sets whose data gets extracted
        workspace - workspace containing the measurements
        settings_group - the settings group used to choose to make the file
        """
        m = workspace.measurements
        acc_file_name = os.path.join(os.path.dirname(self.make_image_file_name(workspace, image_set_numbers[0], settings_group)),ACC_FILE_NAME)
        features = []
        objects_with_selected_features = []
        center_x = ("","")
        center_y = ("","")
        for object_name in object_names:
            if not object_name in m.get_object_names():
                continue
            rfeatures = m.get_feature_names(object_name)
            rfeatures = self.filter_columns(rfeatures, object_name)
            ofeatures = [x for x in rfeatures if not [y for y in REMOVE_FEAT if y in x]]
            ofeatures = [(object_name, feature_name)
                         for feature_name in ofeatures]
            ofeatures.sort()
            features += ofeatures
            # Haggish way to find feature to use as object coordinates
            if ofeatures:
                objects_with_selected_features.append(object_name)
            coord = [feat for feat in rfeatures if "Location_Center_" in feat]
            for feat in coord:
                if (not center_x or "Nuclei" == object_name) and "Center_X" in feat:
                    center_x = (object_name,feat)
                if (not center_y or "Nuclei" == object_name) and "Center_Y" in feat:
                    center_y = (object_name,feat)
        features.insert(0,center_y)
        features.insert(0,center_x)

        # Write ACC file
        try:
            fd = open(acc_file_name,"w")
            for feat in features:
                fd.write(feat[0]+"_"+feat[1]+"\n")
            fd.close()
        except:
            pass

        for img_number in image_set_numbers:
            try:
                file_name = self.make_image_file_name(workspace, img_number, settings_group)
                fd = open(file_name,"w")
                writer = csv.writer(fd,delimiter=DELIMITER)

                object_count =\
                     np.max([m.get_measurement(IMAGE, "Count_%s"%name, img_number)
                             for name in objects_with_selected_features])
                object_count = int(object_count) if object_count else 0
                columns = [np.repeat(img_number, object_count)
                           if feature_name == IMAGE_NUMBER
                           else np.arange(1,object_count+1) 
                           if feature_name == OBJECT_NUMBER
                           else np.repeat(m.get_measurement(IMAGE, feature_name,
                                                            img_number), 
                                          object_count)
                           if object_name == IMAGE
                           else m.get_measurement(object_name, feature_name, 
                                                  img_number)
                           for object_name, feature_name in features]
                for obj_index in range(object_count):
                    row = [ column[obj_index] 
                            if (column is not None and 
                                obj_index < column.shape[0])
                            else np.NAN
                            for column in columns]
                    if self.nan_representation == NANS_AS_NULLS:
                        row = [ 
                            "" if (field is None) or 
                            (np.isreal(field) and not np.isfinite(field))
                            else field for field in row]
                    writer.writerow(row)
                fd.close()
            except:
                pass

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Prepare to create a batch file
        
        This function is called when CellProfiler is about to create a
        file for batch processing. It will pickle the image set list's
        "legacy_fields" dictionary. This callback lets a module prepare for
        saving.
        
        pipeline - the pipeline to be saved
        image_set_list - the image set list to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        
        ExportToACC has to convert the path to file names to
        something that can be used on the cluster.
        '''
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True
            
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        """Adjust the setting values based on the version that saved them
        """
        if variable_revision_number == 1:
            setting_values = (setting_values[:4] + [cps.NO, ""] + setting_values[4:])
            variable_revision_number = 2
        # Standardize input/output directory name references
        SLOT_DIRCHOICE = 1
        directory = setting_values[SLOT_DIRCHOICE]
        directory = cps.DirectoryPath.upgrade_setting(directory)
        setting_values = (setting_values[:SLOT_DIRCHOICE] +
                          [directory] + 
                          setting_values[SLOT_DIRCHOICE+1:])
        
        return setting_values, variable_revision_number, from_matlab
