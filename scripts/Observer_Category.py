import pandas as pd
import numpy as np
from SSN_Config import SSN_ADF_Config as config
from collections import defaultdict
import csv
import matplotlib
from collections import Counter

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

outfile = 'output/observer_categories.csv'

# input_dir = '~/Desktop/Run-2018-6-8'
input_dir = 'output/Run-2018-6-8'
flag_files = {'AO': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('ADF', 'OBS')),
              'AM': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('ADF', 'FULLM')),
              'QO': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('QDF', 'OBS')),
              'QM': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('QDF', 'FULLM'))}

# Size definitions
dpi = 300
pxx = 3000  # Horizontal size of each panel
pxy = pxx  # Vertical size of each panel
frc = 1  # Fraction of the panel devoted to histograms

nph = 1  # Number of horizontal panels
npv = 1  # Number of vertical panels

# Padding
padv = 210  # Vertical padding in pixels
padv2 = 200  # Vertical padding in pixels between panels
padh = 200  # Horizontal padding in pixels at the edge of the figure
padh2 = 200  # Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

# Conversion to relative unites
ppadv = padv / fszv  # Vertical padding in relative units
ppadv2 = padv2 / fszv  # Vertical padding in relative units
ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units


int_flags = True

def make_best_category(outfile, flag_files, r2_threshold=0.05, use_NA_for_overlap=False):
    bases = ['R2']  # Only supports 1 base right now

    # Fields to include in best category output file
    # fields = ['AvThreshold', 'Avg.Res', 'SDThreshold', 'R2OO', 'Avg.ResOO', 'R2DT', 'Avg.ResDT']
    fields = ['QDays', 'ADays', 'NADays', 'TDays', 'QAFrac', 'ObsPerMonth', 'RiseCount', 'DecCount',
              'RiseMonths', 'DecMonths', 'InvInts', 'InvMonths', 'InvMoStreak', 'ObsStartDate', 'ObsTotLength']

    header = ['Observer', 'Flag']
    for b in bases:
        header += [b]
        for f in fields:
            header += ['{}_{}'.format(b, f)]

    obs_cats = defaultdict(dict)

    for FLAG, path in flag_files.items():
        data = pd.read_csv(path, quotechar='"', encoding='utf-8', header=0)
        for r in fields + bases:
            data = data[np.isfinite(data[r])]

        for index, row in data.iterrows():
            obs_id = row['Observer']
            if obs_id in obs_cats.keys():
                for i in range(0, len(bases)):
                    r = bases[i]
                    cur_max_r2 = obs_cats[obs_id][r][1]
                    if row[r] > cur_max_r2 + r2_threshold:
                        fs = [row[r]] + [row[f] for f in fields]
                        obs_cats[obs_id][r] = [FLAG] + fs + [False]
                    elif row[r] == obs_cats[obs_id][r][1] or row[r] > cur_max_r2:
                        if use_NA_for_overlap:
                            obs_cats[obs_id][r][-1] = True
                        else:
                            obs_cats[obs_id][r][0] += '~{}'.format(FLAG)
            else:
                for r in bases:
                    fs = [row[r]] + [row[f] for f in fields]
                    obs_cats[obs_id][r] = [FLAG] + fs + [False]

    writer = csv.writer(open(outfile, 'w'), delimiter=',')

    writer.writerow(header)

    flags_to_int = {'AO': 0, 'AM': 1, 'QO': 0, 'QM': 2, 'GRP': 3}

    group_QO_AO = True
    group_all = True

    exclude_flag = ''

    keys = sorted(obs_cats.keys())
    for key in keys:
        r_dict = obs_cats[key]

        cats = [v[0] if not v[-1] else 'NA' for v in r_dict.values()][0]
        if group_QO_AO and (cats == 'QO~AO' or cats == 'AO~QO' or cats == 'QO'):
            cats = 'AO'

        if group_all and ('~' in cats or 'NA' in cats):
            cats = 'GRP'

        if cats == exclude_flag:
            continue

        if (not use_NA_for_overlap or int_flags) and ('NA' in cats or 'GRP' in cats):
            continue

        if int_flags:
            cats = flags_to_int[cats]


        # sum turns the double list into a single list
        #nums = sum([[round(j, 3) for j in v[1:-1]] if not v[-1] else [0 for _ in v[1:-1]] for v in r_dict.values()], [])
        nums = sum([[round(j, 3) for j in v[1:-1]] for v in r_dict.values()], [])

        r = [key] + [cats] + nums

        writer.writerow(r)


def plot_best(file, vars=('R2', 'R2OO', 'R2DT'), show_plot=True):
    with open(file, 'r') as w:
        # Read csv file with flag data
        rows = []
        reader = csv.reader(w, delimiter=',')
        for r in reader:
            rows.append(r)

        # Set up plot
        if show_plot:
            fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
            ax = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc])
            # ax.set_title('Best flag combinations based on different calculations of R^2')

        Clr = [(0.00, 0.00, 0.00),
               (0.31, 0.24, 0.00),
               (0.43, 0.16, 0.49),
               (0.32, 0.70, 0.30),
               (0.45, 0.70, 0.90),
               (1.00, 0.82, 0.67)]

        flag_cols = {'AM': Clr[4], 'AO': Clr[3],
                     'QM': Clr[2], 'GRP': Clr[5]}

        # Find indices of flag variables in csv file
        inds = {v: rows[0].index(v) - 1 for v in vars}

        # Set up main data storage
        data = []
        labels = []
        obs_dict = defaultdict(list)

        for flag, color in flag_cols.items():
            pts_dict = defaultdict(list)
            labels.append(flag)
            for row in rows[1:]:
                for r2_type, flag_index in inds.items():
                    if flag in row[flag_index]:
                        obs_dict[flag].append(row[0])
                        pts_dict[r2_type].append(float(row[flag_index + 1]))

            # Iterate of pts_dict to make sure data is in right order
            for r2_type in vars:
                pts = pts_dict[r2_type]
                data.append(pts)
            data.append([])

        data = data[:-1]
        labels.append('')
        dlengths = [len(d) for d in data]

        upperLabels = (list(vars) + ['']) * len(flag_cols.keys())

        if show_plot:
            ax.set_ylim(0, 1)
            bot, top = ax.get_ylim()
            numBoxes = len(data)
            pos = np.arange(numBoxes) + 1
            for tick in range(numBoxes):
                ax.text(pos[tick], top - (top * 0.03), upperLabels[tick],
                        horizontalalignment='center', size='small')
                ax.text(pos[tick], top - (top * 0.07),
                        ('{} pts'.format(dlengths[tick]) if dlengths[tick] is not 0 else ''),
                        horizontalalignment='center', size='x-small')

            bplot = ax.boxplot(data, patch_artist=True)

            # box = plt.boxplot(data, notch=True, patch_artist=True)

            colors = ['red', 'lightgreen', 'yellow', (0, 0, 0)] * len(flag_cols.keys())
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        labs = []
        for l in labels:
            labs += ['', l] + [''] * (len(vars) - 1)

        if show_plot:
            ax.set_xticklabels(labs, rotation=45, fontsize=12)
            plt.show()

        return labels, data, obs_dict


def plot_all(flag_files_dict, make_cat_file=True, use_NA_for_overlap=False, r2_threshold=0.05, fit_param='R2'):

    """
    Creates a plot displaying groups of observers for which certain flags perform the best.
    Each group will show performance of each different method on the observers for which
        one of those methods performs the best.
    The method for which observers perform the best in each group will be highlighted.

    :param flag_files_dict: Dictionary of CSV files containing Observer data for each flag in use
    :param make_cat_file: True if should remake the observer_categories file
    :param use_NA_for_overlap: Use NA instead of joining flags when multiple methods perform similarly well
    :param r2_threshold: Include flag in best fit group for an observer when within this threshold of best fit
    :param fit_param: Parameter which determines what variable to base best fit off of
    :return: Displays GUI plot
    """


    # If true, remake best category file
    if make_cat_file:
        make_best_category(outfile, flag_files_dict, use_NA_for_overlap=use_NA_for_overlap, r2_threshold=r2_threshold)

    # Set up plot
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
    ax = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc])
    # Set up plot colors
    alpha = 0.4
    Clr = [(0.00, 0.00, 0.00, alpha),
           (0.31, 0.24, 0.00, alpha),
           (0.43, 0.16, 0.49, alpha),
           (0.32, 0.70, 0.30, alpha),
           (0.45, 0.70, 0.90, alpha),
           (1.00, 0.82, 0.67, alpha)]
    flag_cols = {'AM': Clr[3], 'AO': Clr[4],
                 'QM': Clr[2], 'GRP': Clr[5]}

    # Flags used in each observer group
    OBS_FLAGS = ('AM', 'AO', 'QM')

    # Main data storage
    all_data = {}

    # Read data from flag CSV output files
    for flags, path in flag_files_dict.items():
        data = pd.read_csv(path, quotechar='"', encoding='utf-8', header=0)
        data = data[np.isfinite(data[fit_param])]
        all_data[flags] = data

    # Get best observer data from plot_best method
    labs, data, obs_dict = plot_best(outfile, [fit_param], show_plot=False)
    labs = labs[:-1]

    # List of groups of observers
    OBS_GROUPS = list(obs_dict.keys())

    # Main plot data storage
    all_data = []
    new_labs = []

    # Begin gathering observer data
    for i, f_grp in enumerate(OBS_GROUPS):
        obs_list_f1 = obs_dict[f_grp]
        flag_data = {}

        # Iterate through flag data within each group
        for j, f2 in enumerate(OBS_FLAGS):
            d = {}
            df = pd.read_csv(flag_files_dict[f2], quotechar='"', encoding='utf-8', header=0)
            # Read flag file rows (move this before iterating over groups to decrease work load)
            for index, row in df.iterrows():
                obs_id = int(row['Observer'])
                # If found observer in group and flag file then add to data list
                if str(obs_id) in obs_list_f1 and np.isfinite(row[fit_param]):
                    d[obs_id] = row[fit_param]
                    continue
            # Add all flag data to group data set
            flag_data[f2] = d

        # All data for group f1
        dat = [ob for fd in flag_data.values() for ob in fd.keys()]

        # How many files the observer must be present in in order to be counted
        cnt_threshold = 2
        cnt = Counter(dat)
        # List of observers that have at least 'cnt_threshold' appearances in flag files
        good_obs = [k for k, v in cnt.items() if v >= cnt_threshold]
        # Re-iterate over flags to check for valid observers
        for k, f3 in enumerate(OBS_FLAGS):
            # If observer in flag_data is in good observers list, keep it
            good_data = [r for o, r in flag_data[f3].items() if o in good_obs]
            # Group data to master list (maybe convert to dictionary for simplicity
            all_data.append(good_data)
            new_labs.append(f3)
        # Add empty data after each group for plot spacing
        all_data.append([])
        new_labs.append('')

    # Trim empty data off end
    all_data = all_data[:-1]
    new_labs = new_labs[:-1]

    # Set up axes
    ax.set_ylabel(fit_param)
    ax.set_ylim(0, 1)
    bot, top = ax.get_ylim()
    numBoxes = len(all_data)
    pos = np.arange(numBoxes) + 1

    # Number of data points for each set
    dlengths = [len(d) for d in all_data]

    # Highlighted boxes index
    manual_index = [0, 5, 10]

    # Plot colors
    cols = [flag_cols[l] if l is not '' else (0, 0, 0, 0) for l in new_labs]
    # Add text above boxes
    n = len(OBS_FLAGS)
    for i, f in enumerate(OBS_GROUPS):
        txt = 'Observers best fit with {}'.format(f)
        if f == 'GRP':
            txt = 'Multiple methods'

        ax.text(pos[i * (n + 1) + 1], top - (top * 0.03), txt,
                horizontalalignment='center', size='medium')
        ax.text(pos[i * (n + 1) + 1], top - (top * 0.07),
                ('{} pts'.format(dlengths[i * (n + 1) + 1]) if dlengths[i * (n + 1) + 1] is not 0 else ''),
                horizontalalignment='center', size='small')

    # Main plot method
    bplot = ax.boxplot(all_data, patch_artist=True)

    # Add flag tick labels
    ax.set_xticklabels(new_labs, rotation=45, fontsize=12)

    # Add box colors
    for i, (patch, color) in enumerate(zip(bplot['boxes'], cols)):
        if i in manual_index:
            color = list(color)[:-1] + [1.0]
        patch.set_facecolor(color)

    # Set plot title
    if use_NA_for_overlap:
        ax.set_title('Observers with highlighted flag providing best fit, exclusive flag fit overlap')
    else:
        # ax.set_title('Observers with highlighted flag providing best fit, {}% inclusive flag fit overlap'.format(r2_threshold*100))
        ax.set_title('Performance of ADF calculation methods')

    # Show plot
    plt.show()


################
#     MAIN     #
################
plot_all(flag_files, make_cat_file=True, use_NA_for_overlap=True, r2_threshold=0.1, fit_param='R2')
