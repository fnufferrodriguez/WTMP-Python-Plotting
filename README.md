# WTMP Python Plotting

This repository contains all the Python scripts related to plotting for the Bureau of Reclamation (USBR) Water Temperature Modeling Platform (WTMP) reporting. 
All the code to create the plots and XML files for the Jasper report is contained in this repository. 
This code gets compiled into an executable for the WAT to call to create the reports. 
The code reads in the XML and DSS outputs from the WTMP models and creates all necessary plots and an XML file with the report data for Jasper to read in.
These files are in Python.

The *WAT_Report_Generator.py* acts as the main file in this repository. It calls functions from the other .py files that are grouped together by similar function.
This code is used for both the W2 and ResSim models. Differences between the models are handled within the code.

## Dependencies
All Python dependencies are in the *environment.yml* file. The code is meant to be compiled and ran inside the WAT.


## Building
There are two options for building this package with gradle, using GitHub or a local build. All production and official testing builds are created via the GitHub build system. This requires someone with Reclamation credentials to trigger a build and, if necessary, work through the official build progression to create artificats for every package. Local builds should be used when testing or when builds via GitHub are unavailable.

### Building on GitHub
The build process on GitHub is already preconfigured within GitHub actions. There are two actions available for building, a test build and a publishing build. These differ only in that the publishing build places an artifact on to the GitHub packaging system following the YYYY.MM.DD naming convention. This convention does prevent multiple releases from occuring within a single day. If this becomes an issue, please contact a member of the core Reclamation WTMP development team for assistance. 

The test build is available only for redundancy. Test builds should be done locally to reduce use of limited GitHub Action resources. A passing local test build is a prerequisite for triggering the publishing workflow.

### Building locally

**WARNING: This cannot be built locally on OneDrive or any path containing a space!**

A local build occurs outside of GitHub on a developer's own machine. This is the preferred mechanism for confirming changes are functional, and a local test build should be made and validated before making a pull request. Additionally, a local build may be utilized when GitHub is unavailable. 

The specific configuration needed for local will vary based on the developer's environment and the specific repository being built. These instructions are specified for the Bureau of Reclamation's environment on a Windows machine. This process is intended to conform with Department of Interior security practices and may differ nontrivially in other environments. Producing a local build has the following steps:

1) Installing Java 17.0.x

	The version of Java necessary for gradle is currently misaligned with the version that's being utilized within the WAT itself. This requires downloading a separate version of Java to be able to support the gradle build process. It is recommended that users download and extract OpenJDK 17.0.2 from https://jdk.java.net/archive/. This can be place anywhere on the development machine where the user has read/write/execute permissions outside of the target repository folder.


2) Modifying the user gradle environment

	Modifications to gradle properties is necessary to retarget against the previous Java version and to utilize system certificates. This will require editing a gradle properties file. Gradle will, by default, attempt to utilize whatever Java version is on the machine system path. This version is unlikely to be compatible with either HEC-WAT or Gradle. Additionally, the Reclamation environment automatically blocks the SSL connections needed to obtain upstream dependencies unless SSL certificates are made available to gradle. 

	<br>Within the Windows user folder (e.g. C:/Users/dloney), there should be a .gradle folder (e.g. C:/Users/dloney/.gradle) that contains gradle configuration information. The folder is hidden by default and will require enabling hidden folders/files within Windows Explorer. If the folder does not exist, create it. Similarly, within the .gradle folder, there should be a gradle.properties file (C:/Users/dloney/.gradle/gradle.properties). If one doesn't exist, create it. The .properties should be the extension for the file rather than .txt or another extension.
	
	Open the gradle.properties file and add the following two lines:
	
	```
	org.gradle.java.home=/Path/to/Java 17.0.x
	org.gradle.jvmargs=-Djavax.net.ssl.trustStoreType=WINDOWS-ROOT
	```
	
	The path in the first line should be modified to point to the top level jdk-17.0.x folder at the location from the previous step. The second line does not require any modification.  


3) Running the local build script

	The build_local.bat file should be ran within the command line. This will start the gradle build daemon and begin the build process. If gradle is not currently running in the environment, it may take some time for the script to initialize the daemon. Gradle initialization time should be reduced when building the target or downstream repositories subsequently. 
	
	<br>The end of the build process will either result in a green "BUILD SUCCESSFUL" notification or a red "BUILD FAILED" notification. The build is configured to enable a stack trace on failure to aid in debugging. Resolve any errors and repeat the build until it successfully completes.
	
	<br>It is important to note that the build will execute for whatever branch the repository is currently on. This is also important for downstream repositories that will potentially key off of local packages named by branch. As this repository has no upstream dependencies, this does not apply to this repository. 


4) Publishing the gradle package

	When the gradle build completes, the user may access the build/libs folder within the target repository to obtain the complied executable. This may be sufficient to replace the existing executable within the WAT if one is testing only this repository. However, downstream repositories may require access through gradle to meet their build dependencies lest they default back to GitHub. To do this, one needs to publish the local upstream dependency for gradle to access and update the version number. 
	
	<br>If building downstream repositories, make the local build available to them via the following command from the command prompt located in the target repository:
	
	```
	./gradlew publishToMavenLocal
	```
	
	This will make the package available for downstream packages to access.


5) Stopping gradle 

	By default, gradle will leave the daemon running. This is not necessarily an issue if one is testing in rapid succession or building subsequent repositories. However, this might not be desired in all cases as the daemon consumes a significant amount of memory. To terminate the gradle daemon, open the task manager, find the OpenJDK platform library, and end the task. Force ending the task is not recommended if multiple instances are in use as it is challenging to assign a Java instance to a specific program without additional effort. 

### Building Without Gradle
After making any desired changes to the code, a new executable must be compiled and placed in the WAT build.
1. To create a conda environment, in a command prompt type: `conda env create -f environment.yml`
2. Activate the environment by running the line: `conda activate plotting-env`
3. Create an executable by running the line: `pyinstaller WAT_Report_Generator.spec`
4. In the WAT build *HEC-WAT/AutomatedReport* directory, delete any old files.
5. From the *dist* directory, copy the *WAT_Report_Generator.exe* file and paste them into the WAT build */HEC-WAT/AutomatedReport* directory. This file should be relatively large, approaching about 300 MB in size.