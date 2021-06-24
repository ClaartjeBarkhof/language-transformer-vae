from make_cmaps import make_cmap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set()
import os


if os.path.exists("/home/cbarkhof"):
    user = "cbarkhof"
else:
    user = "ec2-user"

# Not sure how to make this work
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

# --------------------------------------------------------------------------
# FONT
# --------------------------------------------------------------------------


def get_font_prop(f_style):
    font_dict = {
        "medium": f"/home/{user}/code-thesis/NewsVAE/final-analysis/fonts/IBMPlexSans-Medium.ttf",
        "regular": f"/home/{user}/code-thesis/NewsVAE/final-analysis/fonts/IBMPlexSans-Regular.ttf",
        "light": f"/home/{user}/code-thesis/NewsVAE/final-analysis/fonts/IBMPlexSans-ExtraLight.ttf"
    }
    import matplotlib.font_manager as fm
    prop = fm.FontProperties(fname=font_dict[f_style])
    return prop

# --------------------------------------------------------------------------
# COLOUR
# --------------------------------------------------------------------------

# ALL COLOURS

c_dict = {
    "black": "#131313",
    'dark_blue': '#10277C',
    'steal_blue': '#356FB2',
    'bright_blue': '#55B9F9',
    'pink': '#F3B5E0',
    'orange': '#EE6A2C',
    'light_grey': "#E9E9E9"
}

# ---------------------------------------

# CMAP: a colour map from black, darkblue, to bright blue, pink and orange

cmap_c_dict = {
    "black": "#131313",
    'dark_blue': '#10277C',
    'steal_blue': '#356FB2',
    'bright_blue': '#55B9F9',
    'pink': '#F3B5E0',
    'orange': '#EE6A2C',
}
cmap = make_cmap(list(cmap_c_dict.values()), n_colors=1000)

cmap_c_dict_2 = {
    'dark_blue': '#10277C',
    'steal_blue': '#356FB2',
    'bright_blue': '#55B9F9',
    'pink': '#F3B5E0',
    'orange': '#EE6A2C',
}
cmap_2 = make_cmap(list(cmap_c_dict.values()), n_colors=1000)

# ---------------------------------------

# Opt. colours: for plotting different optimisation techniques

opt_c_dict = {
    "MDR": "#356FB2",  # mdr = steal_blue
    "FB": "#EE6A2C",  # fb = orange
    "CYC-FB": "#55B9F9",  # cyc-fb = bright blue
    "CYC": "#C53827",  # cyc = red
    "AE": "#539D66",  # ae = green
    "VAE": "#F3B5E0", #  # vae = pink
    "prior": "#131313" # black
}

mpl.rcParams['axes.facecolor'] = c_dict["light_grey"]