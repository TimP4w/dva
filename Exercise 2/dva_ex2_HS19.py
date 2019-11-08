import numpy as np
import os
import random
from PIL import Image
from bokeh.plotting import figure, curdoc, output_file
from bokeh.layouts import column, row
from bokeh.models import Slider, Select, Button, CustomJS
from bokeh.models import ColumnDataSource
from scipy.ndimage.filters import gaussian_filter, median_filter

# Constants
MEDIAN = "Median"
GAUSS = "Gauss"
ORIGINAL = "Original"
FILTERED = "Filtered"
NOISY = "Noisy"
BLUE_CHANNEL = "Blue Channel"
GREEN_CHANNEL = "Green Channel"
RED_CHANNEL = "Red CHannel"
GREYSCALE = "Greyscale"
GREYSCALE_WEIGHTS = {"red": 0.3, "green": 0.59, "blue": 0.11}
# open and convert image to a usable format
def open_image(name):
    image = Image.open(os.path.abspath(name + ".jpg")).convert("RGBA")
    xdim, ydim = image.size
    orig = np.flipud(np.array(image))
    return orig, [xdim, ydim]


# Extract Red Channel from the image (set other channels to 0)
def red_channel(img):
    image = np.copy(img)
    image[:, :, 1] *= 0
    image[:, :, 2] *= 0
    return image


# Extract Green Channel from the image (set other channels to 0)
def green_channel(img):
    image = np.copy(img)
    image[:, :, 0] *= 0
    image[:, :, 2] *= 0
    return image


# Extract Blue Channel from the image (set other channels to 0)
def blue_channel(img):
    image = np.copy(img)
    image[:, :, 0] *= 0
    image[:, :, 1] *= 0
    return image


# Compute greyscale version by multiplying the different channels and merge them together
def greyscale(img):
    image = np.copy(img)

    red = np.multiply(image[:, :, 0], GREYSCALE_WEIGHTS["red"])
    green = np.multiply(image[:, :, 1], GREYSCALE_WEIGHTS["green"])
    blue = np.multiply(image[:, :, 2], GREYSCALE_WEIGHTS["blue"])

    image[:, :, 0] = red + green + blue
    image[:, :, 1] = red + green + blue
    image[:, :, 2] = red + green + blue
    image[:, :, 3] = 255

    return image


# Add noise to an image
def salt_pepper_noise(img, img_size, percentage):
    image = np.copy(img)
    black = [0, 0, 0, 255]
    white = [255, 255, 255, 255]

    for i in range(int(img_size[0] * img_size[1] / 100 * percentage)):
        black_or_white = random.randint(0, 1)
        x_pixel = random.randint(0, img_size[0] - 1)
        y_pixel = random.randint(0, img_size[1] - 1)

        if black_or_white:
            image[y_pixel, x_pixel, :] = black
        else:
            image[y_pixel, x_pixel, :] = white

    return image


# This function is used to change images. To avoid code duplicates, this function can also
# be called at startup to initialize the dashboard. You should construct the image datasources for all figures in this
# function. Read the assignment 2 slides for tips and implementation suggestions regarding the grayscale image.
def change_image(new):
    img, [xdim, ydim] = open_image(new)
    global img_source, original_img_source, noisy_img_source, red_img_source, green_img_source, blue_img_source, greyscale_img_source, img_size, current_image

    update_source(original_img_source, img)
    update_source(img_source, img)
    update_source(noisy_img_source, img)
    update_source(red_img_source, red_channel(img))
    update_source(green_img_source, green_channel(img))
    update_source(blue_img_source, blue_channel(img))
    update_source(greyscale_img_source, greyscale(img))

    img_size = [xdim, ydim]
    current_image = new


# This function must be triggered each time a different image is chosen. It must call the change_image function with the new
# value passed from the widget and initiate the necessary reset process. (See assignment description for details)
def select_image(attr, old, new):
    change_image(new)
    reset()


# This function must be triggered when the reset button is clicked. You can use change_image to reset the images back to the
# original state by accessing the image select widgets value. Also initiates the necessary reset process.
def reset_dashboard():
    reset()


# Helper function for the reset process. Carefully read in the assignment description which values should be reset and
# what values they should be assigned.
def reset():
    global current_filter, filter_slider, filter_selector

    filter_selector.value = MEDIAN
    current_filter = MEDIAN
    filter_slider.value = 0
    fig1.title.text = ORIGINAL
    noise_slider.value = 0

    global current_image, img_source, noisy_img_source

    img, [xdim, ydim] = open_image(current_image)
    update_source(img_source, img)
    update_source(noisy_img_source, img)


# This function must be triggered when the left mouse button is realeased and the noise slider stops on a new value.
# You should use the passed value as a percentage value and make the necessary calculations. For a fast implementation
# use numpys array slicing as much as possible. A possible way to tackle this task is the following:
# 1. Find out how many pixels you must convert to noise and generate random coordinates within the range of 0 and the
# amount of pixels the image has.
# 2. Flatten your image into a 1 dimensional array of pixelvectors (So technically a 2D array)
# 3. Assign black values to the pixels that are at the first half of the generated coordinates.
# 4. Assign white values to the pixels that are at the second half of the generated coordinates.
# 5. Reshape your array back to 3 dimensions (A 2D array containing pixelvectors)
# Don't forget to assign the new data to the correct sources of the figures
def add_noise(attr, old, new):
    global noisy_img_source, img_source, original_img_source, img_size

    img = get_image_source(original_img_source)
    noisy_image = salt_pepper_noise(img, img_size, new)
    update_source(noisy_img_source, noisy_image)
    update_source(img_source, noisy_image)
    fig1.title.text = NOISY


# Helper function to update the source image of a source
def update_source(source, new_image_source):
    source.data["image"] = [new_image_source]


# Helper function to get the image data source to improve readability
def get_image_source(source):
    return source.data["image"][0]


# This function must be triggered when the filter button is pressed. You can use the scipy Gauss and median filter that
# are imported at the top. Use the value of the filter_slider to determine the filter parameters.
def filter_noise():
    global img_source, original_img_source, current_filter, filter_slider

    img = get_image_source(img_source)

    if filter_slider.value == 0:
        print("Filter not applied, filter value = 0")
        return

    if current_filter == GAUSS:
        filtered_image = gaussian_filter(
            input=img, sigma=(filter_slider.value, filter_slider.value, 0)
        )
    else:
        filtered_image = median_filter(
            input=img, size=(filter_slider.value, filter_slider.value, 1)
        )

    update_source(img_source, filtered_image)

    fig1.title.text = FILTERED


# This function must be triggered when the a different filter is selected in the filter select widget. Use this to
# change the appearance of the filter slider deppending on which filter is chosen.
def change_filter_slider(attr, old, new):
    global current_filter

    if new == GAUSS:
        filter_slider.start = 0
        filter_slider.end = 5
        filter_slider.step = 0.1
        filter_slider.title = "Sigma"

        # Avoid displaying impossible value for sigma
        if filter_slider.value > 5:
            filter_slider.value = 5

        current_filter = GAUSS
    else:
        filter_slider.start = 3
        filter_slider.end = 50
        filter_slider.step = 1
        filter_slider.title = "Mask Size (Pixel)"
        current_filter = MEDIAN


###############################

# In the following part you must setup your data sources, figures and your layout.

# Helper variables to make code more readable
current_image = ""
current_filter = MEDIAN

# The first source is given, setup the rest of them.
img_source = ColumnDataSource(data={"image": [0]})
img_size = [0, 0]

# Rest of them
original_img_source = ColumnDataSource(data={"image": [0]})
blue_img_source = ColumnDataSource(data={"image": [0]})
red_img_source = ColumnDataSource(data={"image": [0]})
green_img_source = ColumnDataSource(data={"image": [0]})
greyscale_img_source = ColumnDataSource(data={"image": [0]})
noisy_img_source = ColumnDataSource(data={"image": [0]})


change_image("image_1")


# Implement your figures in the following part. You can use figure and image arguments that can then be passed to all
# instances for a modular and clean build of your code.

fig_args = {
    "x_range": (0, img_size[0]),
    "y_range": (0, img_size[1]),
    "tools": "pan, wheel_zoom",
}
img_args = {"x": 0, "y": 0, "dw": img_size[0], "dh": img_size[1]}


# The first figure is already setup to show you how the arguments are passed. However, the data source of the image is
# missing and must be filled by you.
fig1 = figure(
    title=ORIGINAL, width=int(img_size[0] / 2), height=int(img_size[1] / 2), **fig_args
)
fig1.image_rgba(image="image", source=img_source, **img_args)

# Depending on how you choose to implement the linking and tool behavior of the rest of the figures you might need a
# second set of figure arguments
fig_args2 = {
    "x_range": fig1.x_range,
    "y_range": fig1.y_range,
    "tools": "",
}


# Implement the rest of the figures in the following part
fig2 = figure(
    title=NOISY, width=int(img_size[0] / 4), height=int(img_size[1] / 4), **fig_args2
)
fig2.image_rgba(image="image", source=noisy_img_source, **img_args)

fig3 = figure(
    title=ORIGINAL, width=int(img_size[0] / 4), height=int(img_size[1] / 4), **fig_args2
)
fig3.image_rgba(image="image", source=original_img_source, **img_args)

fig4 = figure(
    title=RED_CHANNEL,
    width=int(img_size[0] / 4),
    height=int(img_size[1] / 4),
    **fig_args2
)
fig4.image_rgba(image="image", source=red_img_source, **img_args)

fig5 = figure(
    title=GREEN_CHANNEL,
    width=int(img_size[0] / 4),
    height=int(img_size[1] / 4),
    **fig_args2
)
fig5.image_rgba(image="image", source=green_img_source, **img_args)

fig6 = figure(
    title=BLUE_CHANNEL,
    width=int(img_size[0] / 4),
    height=int(img_size[1] / 4),
    **fig_args2
)
fig6.image_rgba(image="image", source=blue_img_source, **img_args)

fig7 = figure(
    title=GREYSCALE,
    width=int(img_size[0] / 4),
    height=int(img_size[1] / 4),
    **fig_args2
)
fig7.image_rgba(image="image", source=greyscale_img_source, **img_args)

# Implement the widgets needed for the interaction with the plots. The reset button is already provided as an example.
# As you can see, the on_click method of the button is used to connect the reset_dashboard function with the clicking
# behavior of the button. Find out how to connect the other widgets to their respective functions that must be triggered
# on some action. Additionally, the js_on_click function is executed when the button is pressed. This allows to have
# separate callbacks in the python code and with javascript on the webpage. This particular javascript function emits
# the reset signal on all plots (same as clicking reset on the toolbar), which resets the zoom and pan of the image.

reset_button = Button(label="Reset", width=int(img_size[0] / 8))
reset_button.on_click(reset_dashboard)
reset_button.js_on_click(
    CustomJS(
        args=dict(p=[fig1, fig2, fig3, fig4, fig5, fig6, fig7]),
        code="""
    for (var i = 0; i < p.length; i++){
        p[i].reset.emit()
    }""",
    )
)


image_selector = Select(
    options=["image_1", "image_2", "image_3", "image_4", "image_5"],
    value="image_1",
    title="Image",
)
image_selector.on_change("value", select_image)


noise_slider = Slider(start=0, end=50, value=0, step=1, title="Noise (%)")
noise_slider.callback_policy = "mouseup"
noise_slider.on_change("value_throttled", add_noise)

filter_slider = Slider(start=3, end=50, value=0, step=1, title="Mask Size (Pixel)")

filter_selector = Select(options=[MEDIAN, GAUSS], value=MEDIAN, title="Filter Type",)
filter_selector.on_change("value", change_filter_slider)


filter_button = Button(label="Filter", width=int(img_size[0] / 8))
filter_button.on_click(filter_noise)
# Use the curdoc function to construct a layout of your dashboard. The column and row functions might also come in
# handy to layout your plots.

menu = column(
    image_selector,
    filter_selector,
    noise_slider,
    filter_slider,
    filter_button,
    reset_button,
)
layout = column(row(fig1, column(fig2, fig3), menu), row(fig4, fig5, fig6, fig7))
curdoc().add_root(layout)
curdoc().title = "dva_ex2"
