import csv
import numpy as np
from SSN_ADF import ssnADF
from SSN_Config import SSN_ADF_Config
import SSN_ADF_Plotter
import argparse
from multiprocessing import Pool
import os
import gc


parser = argparse.ArgumentParser(description="Specify arguments for SSN/ADF config")
parser.add_argument('-Q', "--QDF", help="Run using QDF calculation", action='store_true')
parser.add_argument('-A', "--ADF", help="Run using ADF calculation", action='store_true')
parser.add_argument('-M', "--month", help="Run using full month length in ADF calculation", action='store_true')
parser.add_argument('-O', "--obs", help="Run using total observed days in ADF calculation", action='store_true')
parser.add_argument("-t", "--threads", help="Number of threads to use in multiprocessing", type=int)
parser.add_argument("--start-id", help="ID of the observer to start at", type=int)
parser.add_argument("--end-id", help="ID of the observer to end at", type=int)
parser.add_argument("--skip-observers", help="Ignore observers who have a plot present "
                                             "(see SKIP_OBSERVERS_WITH_PLOTS in config)", action='store_true')
parser.add_argument("--suppress-warnings", help="Suppress all numpy related warnings (use with caution)"
                    , action='store_true')

args, leftovers = parser.parse_known_args()

#################
# CONFIGURATION #
#################

# Observer ID range and who to skip
SSN_ADF_Config.OBS_START_ID = 580
SSN_ADF_Config.OBS_END_ID = 600
SSN_ADF_Config.SKIP_OBS = [356, 417, 423, 424, 531, 576, 579]  # FOR AM
# SSN_ADF_Config.SKIP_OBS = [332, 471, 574, 579]  # FOR AO

# Quantity to use in the numerator of the ADF:  Active days "ADF" or 1-quiet days "QDF"
SSN_ADF_Config.ADF_TYPE = "QDF"

# Quantity to use in the denominator:  Observed days "OBS" or the full month "FULLM"
SSN_ADF_Config.MONTH_TYPE = "OBS"

# Flag to turn on saving of figures
plotSwitch = True

# Output Folder
output_path = 'Run-2018-6-8'

# Output CSV file path
output_csv_file = 'output/{}/{}Observer_ADF.csv'.format(output_path, SSN_ADF_Config.get_file_prepend(
    SSN_ADF_Config.ADF_TYPE, SSN_ADF_Config.MONTH_TYPE))

#####################
# PARSING ARGUMENTS #
#####################
# Arguments will over-ride the options set above

# Quantity to use in the numerator of the ADF:  Active days or 1-quiet days
if args.QDF and args.ADF:
    raise ValueError('Invalid Flags: Can only use one ADF/QDF flag at a time')
elif args.QDF:
    SSN_ADF_Config.ADF_TYPE = "QDF"  # Set to 'QDF' to use 1-QDF calculation.
elif args.ADF:
    SSN_ADF_Config.ADF_TYPE = "ADF"  # Set to 'ADF'  to use ADF calculation.

# Quantity to use in the denominator:  Observed days or the full month
if args.month and args.obs:
    raise ValueError('Invalid Flags: Can only use one FULLM/OBS flag at a time')
elif args.month:
    SSN_ADF_Config.MONTH_TYPE = "FULLM"  # Set to 'FULLM' to use full month length to determine ADF
elif args.obs:
    SSN_ADF_Config.MONTH_TYPE = "OBS"  # Set to 'OBS' to use observed days to determine ADF

# Set number of threads
if args.threads is not None:
    SSN_ADF_Config.PROCESSES = args.threads

# Set start and end ID of observer loop through command line
if args.start_id is not None:
    SSN_ADF_Config.OBS_START_ID = args.start_id
if args.end_id is not None:
    SSN_ADF_Config.OBS_END_ID = args.end_id

# Flag to skip over already processed observers (see config file for more detail)
if args.skip_observers:
    SSN_ADF_Config.SKIP_OBSERVERS_WITH_PLOTS = args.skip_observers

# Suppress numpy warning messages
if args.suppress_warnings:
    SSN_ADF_Config.SUPPRESS_NP_WARNINGS = args.suppress_warnings

if SSN_ADF_Config.SUPPRESS_NP_WARNINGS:
    np.warnings.filterwarnings('ignore')

###################
# STARTING SCRIPT #
###################

print(
    "Starting script with ADF calculation flags: {} / {}\n".format(SSN_ADF_Config.ADF_TYPE, SSN_ADF_Config.MONTH_TYPE))

# Read Data and plot reference search windows, minima and maxima
ssn_adf = ssnADF(ref_data_path='../input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
                 silso_path='../input_data/SN_m_tot_V2.0.csv',
                 obs_data_path='../input_data/GNObservations_JV_V1.22.csv',
                 obs_observer_path='../input_data/GNObservers_JV_V1.22.csv',
                 output_path='output/' + output_path,
                 font={'family': 'sans-serif',
                       'weight': 'normal',
                       'size': 21},
                 dt=14,  # Temporal Stride in days
                 phTol=2,  # Cycle phase tolerance in years
                 thN=100,  # Number of thresholds including 0
                 thI=1,  # Threshold increments
                 plot=plotSwitch)

# Stores SSN metadata set in a SSN_ADF_Class
ssn_data = ssn_adf.ssn_data

# Naming Columns
header = ['Observer',
          'Station',
          'AvThreshold',  # Weighted threshold average based on the nBest matches for all simultaneous fits
          'SDThreshold',  # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits
          'R2',  # R square of the y=x line using a common threshold
          'Avg.Res',  # Mean residual of the y=x line using a common threshold
          'AvThresholdS',  # Weighted threshold average based on the nBest matches for different intervals
          'SDThresholdS',  # Weighted threshold standard deviation based on the nBest matches for different intervals
          'R2S',  # R square of the y=x line for each separate interval
          'Avg.Res.S',  # Mean residual of the y=x line for each separate interval
          'R2DT',  # R square of the y=x line using the average threshold for each interval
          'Avg.ResDT',  # Mean residual of the y=x line using the average threshold for each interval
          'R2OO',  # R square of the y=x line using a common threshold, but only the valid intervals
          'Avg.ResOO',  # Mean residual of the y=x line using a common threshold, but only the valid intervals
          'QDays',  ####### NEW DATA STARTS HERE #######
          'ADays',
          'NADays',
          'TDays',
          'QAFrac',
          'ObsPerMonth',
          'RiseCount',
          'DecCount',
          'RiseMonths',
          'DecMonths',
          'InvInts',
          'InvMonths',
          'InvMoStreak',
          'ObsStartDate',
          'ObsTotLength']


if not os.path.isfile(output_csv_file):
    with open(output_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


# Start pipeline for an individual observer
def run_obs(CalObsID):
    if CalObsID in SSN_ADF_Config.SKIP_OBS:
        return

    print("######## Starting run on observer {} ########\n".format(CalObsID))

    # Processing observer
    obs_valid = ssn_adf.processObserver(ssn_data,  # SSN metadata
                                        CalObs=CalObsID,  # Observer identifier denoting observer to be processed
                                        MoLngt=30,  # Duration of the interval ("month") used to calculate the ADF
                                        minObD=0.33,
                                        # Minimum proportion of days with observation for a "month" to be considered valid
                                        vldIntThr=0.33)  # Minimum proportion of valid "months" for a decaying or raising interval to be considered valid

    # Continue only if observer has valid intervals
    if obs_valid:

        # Plot active vs. observed days
        if plotSwitch and SSN_ADF_Config.PLOT_ACTIVE_OBSERVED:
            SSN_ADF_Plotter.plotActiveVsObserved(ssn_data)

        # Calculating the Earth's Mover Distance using sliding windows for different intervals
        obs_ref_overlap = ssn_adf.ADFscanningWindowEMD(ssn_data, nBest=50)  # Number of top best matches to keep

        if plotSwitch:
            # Plot active vs. observed days
            if SSN_ADF_Config.PLOT_OPTIMAL_THRESH:
                SSN_ADF_Plotter.plotOptimalThresholdWindow(ssn_data)

            # Plot Distribution of active thresholds
            if SSN_ADF_Config.PLOT_DIST_THRESH_MI:
                SSN_ADF_Plotter.plotDistributionOfThresholdsMI(ssn_data)

            # If there is overlap between the observer and reference plot the y=x scatterplots
            if SSN_ADF_Config.PLOT_INTERVAL_SCATTER and obs_ref_overlap and np.sum(ssn_data.vldIntr) > 1:
                SSN_ADF_Plotter.plotIntervalScatterPlots(ssn_data)

        # Calculating the Earth's Mover Distance using common thresholds for different intervals
        plot_obs = ssn_adf.ADFsimultaneousEMD(ssn_data,
                                              disThres=1.20,
                                              # Threshold above which we will ignore timeshifts in simultaneous fit
                                              MaxIter=1000)
        # Maximum number of iterations above which we skip simultaneous fit

        if plotSwitch and plot_obs:

            if np.sum(ssn_data.vldIntr) > 1:
                # Plotting minimum EMD figure
                if SSN_ADF_Config.PLOT_MIN_EMD:
                    SSN_ADF_Plotter.plotMinEMD(ssn_data)

                # Plot the result of simultaneous fit
                if SSN_ADF_Config.PLOT_SIM_FIT:
                    SSN_ADF_Plotter.plotSimultaneousFit(ssn_data)

                # Plot the distribution of thresholds
                if SSN_ADF_Config.PLOT_DIST_THRESH:
                    SSN_ADF_Plotter.plotDistributionOfThresholds(ssn_data)

            # If there is overlap between the observer and reference plot the y=x scatterplots
            if obs_ref_overlap:
                if SSN_ADF_Config.PLOT_SINGLE_THRESH_SCATTER:
                    SSN_ADF_Plotter.plotSingleThresholdScatterPlot(ssn_data)
                if SSN_ADF_Config.PLOT_MULTI_THRESH_SCATTER:
                    SSN_ADF_Plotter.plotMultiThresholdScatterPlot(ssn_data)

        # Saving row
        y_row = [ssn_data.CalObs,
                 ssn_data.NamObs,
                 ssn_data.wAv,  # Weighted threshold average based on the nBest matches for all simultaneous fits
                 ssn_data.wSD,
                 # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits
                 ssn_data.rSq,  # R square of the y=x line using a common threshold
                 ssn_data.mRes,  # Mean residual of the y=x line using a common threshold
                 ssn_data.wAvI,  # Weighted threshold average based on the nBest matches for different intervals
                 ssn_data.wSDI,
                 # Weighted threshold standard deviation based on the nBest matches for different intervals
                 ssn_data.rSqI,  # R square of the y=x line for each separate interval
                 ssn_data.mResI,  # Mean residual of the y=x line for each separate interval
                 ssn_data.rSqDT,  # R square of the y=x line using the average threshold for each interval
                 ssn_data.mResDT,  # Mean residual of the y=x line using the average threshold for each interval
                 ssn_data.rSqOO,  # R square of the y=x line using a common threshold, but only valid intervals
                 ssn_data.mResOO,  # Mean residual of the y=x line using a common threshold, but only valid intervals
                 ssn_data.QDays,  ####### NEW DATA STARTS HERE #######
                 ssn_data.ADays,
                 ssn_data.NADays,
                 ssn_data.TDays,
                 ssn_data.QAFrac,
                 ssn_data.ObsPerMonth,
                 ssn_data.RiseCount,
                 ssn_data.DecCount,
                 ssn_data.RiseMonths,
                 ssn_data.DecMonths,
                 ssn_data.InvInts,
                 ssn_data.InvMonths,
                 ssn_data.InvMoStreak,
                 ssn_data.ObsStartDate,
                 ssn_data.ObsTotLength
                 ]

        with open(output_csv_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(y_row)

    gc.collect()


if __name__ == '__main__':

    obs_range = range(SSN_ADF_Config.OBS_START_ID, SSN_ADF_Config.OBS_END_ID)

    # If SKIP_OBSERVERS_WITH_PLOTS is set, for each observer find matching directory
    # If there is a file with the same flags as currently set, add this observer to list of observers to skip
    if SSN_ADF_Config.SKIP_OBSERVERS_WITH_PLOTS:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', output_path)
        for ob in obs_range:
            for dname in os.listdir(out_dir):
                if dname.startswith('{}_'.format(ob)):
                    for file in os.listdir(os.path.join(out_dir, dname)):
                        if file.startswith(
                                SSN_ADF_Config.get_file_prepend(SSN_ADF_Config.ADF_TYPE,
                                                                SSN_ADF_Config.MONTH_TYPE)):
                            SSN_ADF_Config.SKIP_OBS.append(ob)
                            break
        print("\nSkipping observers who have plot(s) labeled with the current flags ({} / {}):\n{}\n"
              "Change the SKIP_OBSERVERS_WITH_PLOTS config flag to remove this behavior\n".format(
              SSN_ADF_Config.ADF_TYPE, SSN_ADF_Config.MONTH_TYPE, SSN_ADF_Config.SKIP_OBS))

    obs_range = [x for x in obs_range if x not in SSN_ADF_Config.SKIP_OBS]

    if SSN_ADF_Config.PROCESSES == 1:
        for i in obs_range:
            run_obs(i)  # Start process normally
    elif SSN_ADF_Config.PROCESSES == -1:
        try:
            pool = Pool()  # Create a multiprocessing Pool with all available cores
            pool.map(run_obs, obs_range)  # process all observer iterable with pool
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
    else:
        # Safety checks for process number
        if SSN_ADF_Config.PROCESSES < -1 or SSN_ADF_Config.PROCESSES == 0:
            raise ValueError(
                "Invalid processes number ({}). Please set to a valid number.".format(SSN_ADF_Config.PROCESSES))
        elif SSN_ADF_Config.PROCESSES > os.cpu_count():
            raise ValueError("Processes number higher than CPU count. "
                             "You tried to initiate {} processes, while only having {} CPU's.".format(
                SSN_ADF_Config.PROCESSES, os.cpu_count()))
        try:
            pool = Pool(processes=SSN_ADF_Config.PROCESSES)  # Create a multiprocessing Pool
            pool.map(run_obs, obs_range)  # process all observer iterable with pool
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
