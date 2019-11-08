import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import os
from scipy import interpolate
from bokeh.layouts import layout
from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.palettes import RdYlBu
from bokeh.transform import dodge
from numpy.polynomial.polynomial import polyvander

# CONSTANTS
MAXIMUM_YEAR = 2016
MINIMUM_VALUE = 10


# read data from .csv file by using absolute path
__file__ = "20151001_hundenamen.csv"
data_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
try:
    df1 = pd.read_csv(os.path.join(data_absolute_dirpath, __file__))
except FileNotFoundError:
    print(
        "Couldn't find the dataset file, please check that you have the file in the same folder as the script"
    )
    exit()
except:
    print("Something went wrong while opening the dataset file...")
    exit()

# rename the columns of the data frame
df1.rename(
    columns={
        "HUNDENAME": "name",
        "GEBURTSJAHR_HUND": "birth_year",
        "GESCHLECHT_HUND": "gender",
    },
    inplace=True,
)

# Count the nr of births per year
nr_of_births_per_year = df1.groupby("birth_year").size()

# ====================================================================
# =============== 1. data cleaning and basic plotting ================
# ====================================================================

# task 1.1: Remove outliers and construct a ColumnDataSource from the clean DataFrame
# hint: use df.loc to remove the outliers as specified in the assignment document
# then reset the index of DataFrame before constructing ColumnDataSource
# Only use years from before 2016 and which have more than 10 births per year
# reference dataframe: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# reference columndatasource: https://bokeh.pydata.org/en/latest/docs/reference/models/sources.html

df1 = df1.loc[df1["birth_year"] < MAXIMUM_YEAR]
nr_of_births_per_year = df1.groupby("birth_year").size()[lambda x: x > MINIMUM_VALUE]
df = pd.DataFrame(
    {"Years": nr_of_births_per_year.index, "Numbers": nr_of_births_per_year.values}
)

clean_column_data_source = ColumnDataSource(
    dict(x=df["Years"], y=df["Numbers"], sizes=df["Numbers"] / 20)
)


# task 1.2: construct an array of evenly spaced numbers with a stepsize of 0.1. Start and stop are the earliest and
# latest year contained in your cleaned source data. These will be your x values for the plotting of the interpolation.
# hint: use numpys linspace() function for this task
# the array should look similar to this: array([start, ... , 1999.1, 1999.2, ... , 2014.7, 2014.8, ... , end])
start_year = df["Years"].iloc[0]
end_year = df["Years"].iloc[-1] + 0.1
x_values = np.arange(start_year, end_year, 0.1, float)


# task 1.3: configure mouse hover tool
# reference: https://bokeh.pydata.org/en/latest/docs/user_guide/categorical.html#hover-tools
# your tooltip should contain 'Year' and 'Number' Take care, that only the diamond plot is affected by the hover tool.
tooltip = [("Year", "@x"), ("Number", "@y")]

# task 1.4: generate the figure for plot 1 and add a diamond glyph renderer with a size, color, and alpha value
# reference: https://bokeh.pydata.org/en/latest/docs/reference/plotting.html
# examples: https://bokeh.pydata.org/en/latest/docs/user_guide/plotting.html
# hint: For the figure, set proper values for x_range and y_range, x and y labels, plot_height, plot_width and
# title and remember to add the hovertool. For the diamond glyphs, set preferred values for size, color and alpha
# optional task: set the size of the glyphs such that they adapt it according to their 'Numbers' value
plot_1 = figure(
    plot_width=1500,
    plot_height=500,
    title="Number of dog births per year",
    tools="hover",
    tooltips=tooltip,
)
plot_1.xaxis.axis_label = "Year"
plot_1.yaxis.axis_label = "Number of Dogs"

plot_1.diamond(
    x="x",
    y="y",
    source=clean_column_data_source,
    size="sizes",
    color="navy",
    alpha=0.8,
    legend="Number of dog births per year",
)

# task 1.5: generate the figure for plot 2 with proper settings for x_range, x and y axis label, plot_height,
# plot_width and title
plot_2 = figure(plot_width=1500, plot_height=200, title="Error bars")
plot_2.yaxis.axis_label = "Fitting Error"

# ======================================================================
# ============= 2. Plotting fitting curves and error bars ==============
# ======================================================================

# task 2.1: Perform a piecewise linear interpolation
# hint: this can be achieved in two ways: either with scipys interp1d solution or with a bokeh line plot
plot_1.line(
    x="x",
    y="y",
    source=clean_column_data_source,
    line_width=2,
    color="green",
    alpha=0.8,
    legend="Piecewise linear interpolation",
)

# task 2.2: draw fitting lines in plot 1 and error bars in plot 2 using the following for loop. You should fit curves
# of degree 2, 4 and 6 to the points. The range of the for loop is already configured this way.
for i in range(2, 7, 2):
    # degree of each polynomial fitting
    degree = i
    # color used in both fitting curves and error bars
    color = RdYlBu[11][4 + degree]
    # hint: use numpys polyfit() to calculate fitting coefficients and polyval() to calculate estimated y values for
    # the x values you constructed before.
    fitting_coefficient = np.polyfit(df["Years"], df["Numbers"], i)
    fitting_curve = np.polyval(fitting_coefficient, x_values)

    # hint: construct new ColumnDataSource for fitting curve, x should be the constructed x values and
    # y should be the estimated y. Then draw the fitting line into plot 1, add proper legend, color, line_width and
    # line_alpha
    fitting_curve_column_data_source = ColumnDataSource(
        dict(x=x_values, y=fitting_curve)
    )
    legend = "Polynomial Least-Squares Interpolation: Fitting Degree = " + str(i)
    plot_1.line(
        x="x",
        y="y",
        source=fitting_curve_column_data_source,
        line_width=2,
        line_alpha=0.8,
        color=color,
        legend=legend,
    )

    # draw the error bars into plot 2
    # hint: calculate the fitting error for each year by subtracting the original 'Numbers' value off your cleaned
    # source from the estimated y values. Be careful to match the correct y estimation to the respective 'Numbers'
    # value! For the subsampling look up array slicing for numpy arrays. Use the absolute values of the errors to only
    # get error bars above the baseline.
    y_estimate = fitting_curve[0 : len(x_values) : 10]
    error = abs(df["Numbers"] - y_estimate)
    error_data_source = ColumnDataSource(dict(x=df["Years"], y=error))

    # hint: before plotting, make sure the bars don't overlap each other, i.e. slightly adjust the x position for each
    # bar within each loop cycle
    x_pos = 0.1 * (i / 2 - 1)
    plot_2.vbar(
        x=dodge("x", x_pos),
        top="y",
        source=error_data_source,
        width=0.09,
        alpha=0.8,
        color=color,
    )
# task 2.3: draw a 6th degree smooth polynomial interpolation into plot 1
# hint 1: since most good math libraries use the least square method or similarly stable interpolation approaches you
# have to do this manually. Have a look at the lecture slides DVA 02 page 22 and use numpys (multi dimensional) array
# functions to solve the equations. However, if you find a library/package that calculates the interpolation with the
# method shown in the lecture, you are welcome to use it.
# hint 2: Use the entries 0, 3, 6, 9, 12, 15 of the original column data source

# start by constructing your x and y values from the source
degree = 6
x_val = clean_column_data_source.data["x"][0:16:3]
y_val = clean_column_data_source.data["y"][0:16:3]

# generate a 6x6 matrix filled with 1
V = np.ones((6, 6))
# use array slicing and the power function to fill columns 2-6 of the matrix with the correct values
for i in range(0, degree - 1):
    V[:, i] = np.power(x_val, degree - 1 - i)


# solve the equation from the slides with the correct numpy functions
coefficients = np.linalg.solve(V, y_val)
# use polyval and the x values from task 2.1 together with the calculated coefficients to estimate y values and then
# plot the result into plot 1
smooth_fitting_curve_y = np.polyval(coefficients, x_values)


smooth_fitting_curve = ColumnDataSource(dict(x=x_values, y=smooth_fitting_curve_y))
plot_1.line(
    x="x",
    y="y",
    source=smooth_fitting_curve,
    legend="Smooth Polynomial Interpolation, Fitting Degree = 6",
    color="blue",
)
# draw the error bars for this polynomial again into plot 2
y_estimate = smooth_fitting_curve_y[0 : len(x_values) : 10]
error = abs(df["Numbers"] - y_estimate)
error_data_source = ColumnDataSource(dict(x=df["Years"], y=error))

# move the x positions such that the new bars are to the right of the previous ones
x_pos = 0.1 * (8 / 2 - 1)
plot_2.vbar(
    x=dodge("x", x_pos),
    top="y",
    source=error_data_source,
    width=0.09,
    alpha=0.8,
    color="blue",
)

# set up the position of legend in plot 1 as you like
plot_1.legend.location = "top_left"

# ==============================================
# ================= dashboard ==================
# ==============================================

# put all the plots into one layout
# reference: https://bokeh.pydata.org/en/latest/docs/user_guide/layout.html
# fill in the function


dashboard = layout([[plot_1], [plot_2]])

show(dashboard)
