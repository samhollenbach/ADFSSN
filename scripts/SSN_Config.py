import os


class SSN_Data:
    """
    Blank class for managing SSN data.
    Can assign named variables to this class to avoid global class variables
    """
    pass


class SSN_ADF_Config:
    """
    Class to store static config variables
    """

    # MULTIPROCESSING
    # 1 --> do not use any parallel processing.
    # -1 -->  use all cores on machine.
    # Other --> defines number of cores to use
    PROCESSES = 1

    # OVERWRITING AND SKIPPING PLOTS
    # Setting both flags to false will recreate and overwrite all plots for all observers
    # Overwrite plots already present
    # Even when false, still have to process each observer
    # Safer than the SKIP_OBSERVERS_WITH_PLOTS flag
    SKIP_PRESENT_PLOTS = True
    # Ignore any observer that has any plots with current flags in its output folder
    # Skips processing observer data making the process much faster
    # However, if a plot that should have been made previously is missing it will not be made when this flag is enabled
    # More dangerous than SKIP_PRESENT_PLOTS, but good when confident that existing observers were completely processed
    SKIP_OBSERVERS_WITH_PLOTS = False

    # Plotting config variables
    PLOT_OPTIMAL_THRESH = True
    PLOT_ACTIVE_OBSERVED = True
    PLOT_DIST_THRESH_MI = True
    PLOT_INTERVAL_SCATTER = True
    PLOT_MIN_EMD = True
    PLOT_SIM_FIT = True
    PLOT_DIST_THRESH = True
    PLOT_SINGLE_THRESH_SCATTER = True
    PLOT_MULTI_THRESH_SCATTER = True

    # Suppress numpy warnings for cleaner console output
    SUPPRESS_NP_WARNINGS = True

    @staticmethod
    def get_file_prepend(adf_type, month_type):
        """
        :param adf_type: ADF parameter set in config
        :param month_type: month length parameter set in config
        :return: prepend for plots depending on ADF and month length
        """
        if adf_type == "ADF":
            prepend = "A_"
        elif adf_type == "QDF":
            prepend = "Q_"
        else:
            raise ValueError('Invalid flag: Use \'ADF\' (or \'QDF\') for active (1-quiet) day fraction.')

        if month_type == "FULLM":
            prepend += "M_"
        elif month_type == "OBS":
            prepend += "O_"
        else:
            raise ValueError(
                'Invalid flag: Use \'OBS\' (or \'FULLM\') to use observed days (full month length) to determine ADF.')
        return prepend

    @staticmethod
    def get_file_output_string(number, title, ssn_data, adf_type, month_type):
        """
        :param number: Plot type identifier
        :param title: Plot title
        :param ssn_data: SSN_Data object storing metadata
        :param adf_type: ADF parameter set in config
        :param month_type: month length parameter set in config
        :return: Path
        """
        return os.path.join(ssn_data.output_path,
                            "{}_{}".format(ssn_data.CalObs, ssn_data.NamObs),
                            "{}{}_{}_{}_{}.png".format(SSN_ADF_Config.get_file_prepend(adf_type, month_type),
                                                       number,
                                                       ssn_data.CalObs,
                                                       ssn_data.NamObs,
                                                       title))
