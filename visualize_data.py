import numpy as np
from numpy import linalg
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# Print out a representation of each image within the data


def display_image():

    # How many images to print?
    k = int(input("How many images to print?\n"))

    with open('MNIST_data/mnist_test.csv', 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        next(csvreader)

        counter = 0

        for data in csvreader:
            # Manual stop after k images
            if counter == k:
                break

            # The first column is the label
            label = data[0]

            # The rest of columns are pixels
            pixels = data[1:]

            # Make those columns into a array of 8-bits pixels
            # This array will be of 1D with length 784
            # The pixel intensity values are integers from 0 to 255
            pixels = np.array(pixels, dtype='int64')

            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = pixels.reshape((28, 28))

            # Plot
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(pixels, cmap='gray')
            plt.show()

            counter += 1

    csv_file.close()


# Print out a chart of the pixel numbers


def display_pixel_data():

    unique_values = []

    with open('MNIST_data/mnist_test.csv', 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        next(csvreader)

        for data in csvreader:
            pixels = data[1:]
            pixels = [int(i) for i in pixels]
            for n in pixels:
                unique_values.append(n)

    csv_file.close()

    unique_values = np.array(unique_values, dtype='int64')
    x, height = np.unique(unique_values, return_counts=True)

    # Plot
    plt.title("Pixel Value Count")
    plt.bar(x, height)
    plt.show()


# returns 10x 2D arrays of a list of pixels for each case sorted per label digit 0-9


def seperate_by_digit():

    zero = []
    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []
    nine = []

    with open('MNIST_data/mnist_test.csv', 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        next(csvreader)

        for data in csvreader:

            # The first column is the label
            label = int(data[0])

            # The rest of columns are pixels
            pixels = data[1:]
            pixels = [int(i) for i in pixels]

            if label == 0:
                zero.append(pixels)
            elif label == 1:
                one.append(pixels)
            elif label == 2:
                two.append(pixels)
            elif label == 3:
                three.append(pixels)
            elif label == 4:
                four.append(pixels)
            elif label == 5:
                five.append(pixels)
            elif label == 6:
                six.append(pixels)
            elif label == 7:
                seven.append(pixels)
            elif label == 8:
                eight.append(pixels)
            elif label == 9:
                nine.append(pixels)

    csv_file.close()

    return zero, one, two, three, four, five, six, seven, eight, nine


# returns 10x numpy arrays of the averaged images for each pixel per digit 0-9 (centroids)


def find_centroid():

    zero, one, two, three, four, five, six, seven, eight, nine = seperate_by_digit()

    # Convert python lists into numpy arrays
    zero = np.mean(zero, axis=0)
    one = np.mean(one, axis=0)
    two = np.mean(two, axis=0)
    three = np.mean(three, axis=0)
    four = np.mean(four, axis=0)
    five = np.mean(five, axis=0)
    six = np.mean(six, axis=0)
    seven = np.mean(seven, axis=0)
    eight = np.mean(eight, axis=0)
    nine = np.mean(nine, axis=0)

    return zero, one, two, three, four, five, six, seven, eight, nine


# Print out a representation of averaged images for each digit (centroids)


def display_centroid():

    zero, one, two, three, four, five, six, seven, eight, nine = find_centroid()

    # Reshape the array into 28 x 28 array (2-dimensional array)
    zero = zero.reshape((28, 28))
    one = one.reshape((28, 28))
    two = two.reshape((28, 28))
    three = three.reshape((28, 28))
    four = four.reshape((28, 28))
    five = five.reshape((28, 28))
    six = six.reshape((28, 28))
    seven = seven.reshape((28, 28))
    eight = eight.reshape((28, 28))
    nine = nine.reshape((28, 28))

    # Plot
    f, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(
        nrows=2, ncols=5, sharex=True, sharey=True)
    ax = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    ax0.imshow(zero, cmap='gray')
    im = ax1.imshow(one, cmap='gray')
    ax2.imshow(two, cmap='gray')
    ax3.imshow(three, cmap='gray')
    ax4.imshow(four, cmap='gray')
    ax5.imshow(five, cmap='gray')
    ax6.imshow(six, cmap='gray')
    ax7.imshow(seven, cmap='gray')
    ax8.imshow(eight, cmap='gray')
    ax9.imshow(nine, cmap='gray')

    plt.colorbar(im, ax=ax)

    plt.show()


def euclidean_distance(x, y):
    return np.sqrt(np.mean(np.square(np.subtract(x, y))))


# Print out a representation of the euclidean distance from the digit centroid


def display_centroid_euclidean_distance():

    zero_c, one_c, two_c, three_c, four_c, five_c, six_c, seven_c, eight_c, nine_c = find_centroid()
    nums_c = [zero_c, one_c, two_c, three_c, four_c,
              five_c, six_c, seven_c, eight_c, nine_c]

    zero, one, two, three, four, five, six, seven, eight, nine = seperate_by_digit()
    nums = [zero, one, two, three, four, five, six, seven, eight, nine]

    zero_x = []
    one_x = []
    two_x = []
    three_x = []
    four_x = []
    five_x = []
    six_x = []
    seven_x = []
    eight_x = []
    nine_x = []

    # Find Euclidean Distance
    for i in range(10):
        for num in nums[i]:
            case = np.array(num, dtype='int64')
            dist = euclidean_distance(case, nums_c[i])
            if i == 0:
                zero_x.append(dist)
            elif i == 1:
                one_x.append(dist)
            elif i == 2:
                two_x.append(dist)
            elif i == 3:
                three_x.append(dist)
            elif i == 4:
                four_x.append(dist)
            elif i == 5:
                five_x.append(dist)
            elif i == 6:
                six_x.append(dist)
            elif i == 7:
                seven_x.append(dist)
            elif i == 8:
                eight_x.append(dist)
            elif i == 9:
                nine_x.append(dist)

    # Plot
    plt.title("Euclidian Distance from Centroid")
    box = [zero_x, one_x, two_x, three_x, four_x,
           five_x, six_x, seven_x, eight_x, nine_x]
    columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.boxplot(box, labels=columns)
    plt.show()


display_image()
display_pixel_data()
display_centroid()
display_centroid_euclidean_distance()
