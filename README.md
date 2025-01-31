# Sunspot Series Reconstruction Repository

Welcome to the repository for different bits and pieces of code assembled with the aim of achieving the best possible Sunspot Group Series reconstructions and the collection of different techniques and approaches.

## Setting Up Your Modeling Environment ##

Requirements:

* Anaconda Python 3 installation.
* pyemd package.

### Anaconda Installation ###

You can find instructions for anaconda installation here:

* [Anaconda Windows installation instructions.](https://docs.anaconda.com/anaconda/install/windows)
* [Anaconda Mac-OS installation instructions.](https://docs.anaconda.com/anaconda/install/mac-os)
* [Anaconda Linux installation instructions.](https://docs.anaconda.com/anaconda/install/linux)

### Updating Conda Environment and Installing Pyemd ###

<a id='terminal'></a>
1. Start a terminal.  You can find instructions for opening terminals here:

  * [Open the Command Prompt program in Windows.](http://www.wikihow.com/Open-the-Command-Prompt-in-Windows)
  * [Start a temrminal in Mac-OS.](http://www.wikihow.com/Open-a-Terminal-Window-in-Mac)
  
2. Run the following commands.  They will update your python environment and install Pyemd.

> conda update --all  
> conda install astropy   
> pip install pyemd 

### Clone or Download this Repository ###

Click on the _Clone or download_ button and clone it to a repository or download it as a zip file.

![GitHub Logo](https://help.github.com/assets/images/help/repository/clone-repo-clone-url-button.png)

If you want help cloning the repository and don't know how to do it let me know.  Otherwise, simply download the files and unpack them on a folder in your computer.


## Running the Jupyter Notebooks ##

1. Open a terminal ([see above](#terminal)).

2. Navigate to the folder where you unzipped the repository.  Here are some quick tutorials to use the terminal:
  * [Using the terminal in Windows](https://www.digitalcitizen.life/command-prompt-how-use-basic-commands)
  * [Using the terminal in Mac-OS/Linux](https://computers.tutsplus.com/tutorials/navigating-the-terminal-a-gentle-introduction--mac-3855)
  
3. Run the following command:

> jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000

The _jupyter notebook_ command is normally sufficient, but the visualizations I'm including have a data rate limit to high for the defaults.  There is a way of fixing this permanently, but you need to mess up with configuration files.  More here:

* [Creating a configuration for jupyter](https://jupyter-notebook.readthedocs.io/en/latest/config.html)
* [Common Jupyter directories and file locations](https://jupyter.readthedocs.io/en/latest/projects/jupyter-directories.html)

4. Open the notebook.  Launching jupyter should open the dashboard in your folder.  Something that looks like this:

![Jupyter Dashboard](https://jupyter.readthedocs.io/en/latest/_images/tryjupyter_file.png)

Click on the notebook you want to open.

5. Run all cells or the cells you want.  For more information about running cells please see:

* [Running Jupyter Code](https://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Running%20Code.html)


## Jupyter Notebook Documentation: ##

More information about the Jupyter notebook can be found here:

* [Jupyter notebook documentation](https://jupyter-notebook.readthedocs.io/en/latest/index.html)

## Running the Scripts: ##

When running the scripts with parallel processing enabled, some system encounter the following error:

```
The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.
```

To fix this issue, find your _matplotlibrc_ file by going into the python environment you are using and running the commands:
```python
>>> import matplotlib
>>> matplotlib.matplotlib_fname()
'/home/foo/.config/matplotlib/matplotlibrc' # This output shows where your matplotlibrc file is located
```

Find the line in this file where 'backend' is defined; it should look like this.
```
backend      : TkAgg
```
Comment out this line with a leading '#'. This change sets the matplotlib backend to use the default non-interactive  version.
Interactive versions of the matplotlib backend (like the Tkinter based backend shown above) can cause issues with multiprocessing.