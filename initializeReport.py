'''
Created on 6/9/2021
@author: scott
@organization: Resource Management Associates
@contact: scott@rmanet.com
@note:
rem %1 the watershed folder
rem %2 the simulation folder      <sim dir>
rem %3 model name ...ie. ResSim   <modelname>
rem %4 alternative name           <alt name>
rem %5 obs data folder            <obs dir>
rem %6 region name ...ie.ShastaRes<reg name>

python %1\scripts\initializeReport.py %2 %4

'''

import shutil
import os,sys
import datetime as dt
import XMLReport


# sim_folder = sys.argv[1]
# _, sim_name = os.path.split(sim_folder)
# alt_name = sys.arvg[2]

def clean_output_dir(dir_name):
    files_in_directory = os.listdir(dir_name)
    filtered_files = [file for file in files_in_directory if file.endswith(".png")]
    for file in filtered_files:
        path_to_file = os.path.join(dir_name, file)
        os.remove(path_to_file)


##### TESTING ######
sim_folder = r'Z:\USBR\test'
alt_name = 'Test_init'
_, sim_name = os.path.split(sim_folder)



xml_template = 'report_template.xml'
#copy xml template
# new_xml = os.path.join(sim_folder, xml_template.replace('template', '{0}_{1}'.format(sim_name, alt_name)))
# new_xml = xml_template.replace('template', '{0}_{1}'.format(sim_name, alt_name))
new_xml = 'USBRAutomatedReportOutput.xml'
shutil.copyfile(xml_template, new_xml)

report_date = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
report_name = " : ".join([sim_name, alt_name])

XML = XMLReport.makeXMLReport(new_xml)
XML.writeCover(report_name, report_date)

clean_output_dir('..\Images')

