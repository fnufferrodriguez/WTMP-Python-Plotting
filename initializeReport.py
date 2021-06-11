'''
Created on 6/9/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
rem %1 the watershed folder


'''

import shutil
import os,sys
import datetime as dt
import XMLReport



def clean_output_dir(dir_name, filetype):
    files_in_directory = os.listdir(dir_name)
    filtered_files = [file for file in files_in_directory if file.endswith(filetype)]
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        os.remove(path_to_file)


##### TESTING ######
# sim_folder = r'Z:\USBR\test'
# alt_name = 'Test_init'
# _, sim_name = os.path.split(sim_folder)



xml_template = 'report_template.xml'
#copy xml template
# new_xml = os.path.join(sim_folder, xml_template.replace('template', '{0}_{1}'.format(sim_name, alt_name)))
# new_xml = xml_template.replace('template', '{0}_{1}'.format(sim_name, alt_name))
new_xml = 'USBRAutomatedReportOutput.xml'
shutil.copyfile(xml_template, new_xml)

report_date = dt.datetime.now().strftime('%Y-%m-%d %H:%M')

XML = XMLReport.makeXMLReport(new_xml)
XML.writeCover(report_date)

output_path = '.'
images_path = os.path.join(output_path, '..', "Images")
if not os.path.exists(images_path):
    os.makedirs(images_path)

clean_output_dir('..\Images', '.png')

csv_path = os.path.join(output_path,  "CSV")
if not os.path.exists(images_path):
    os.makedirs(images_path)
clean_output_dir('..\Images', '.csv')