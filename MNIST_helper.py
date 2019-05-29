from matplotlib import pyplot as plt
import numpy as np

def plot_number(x_train, y_train, number, show_label=True, figsize=(10, 5)):
    plt.imshow(x_train[number], cmap='gray')
    if show_label:
        plt.text(0,0,str(y_train[number]), color='w', size=20, verticalalignment="top")
    plt.show()
    
def create_row(x_train, numbers):
    concatenated = x_train[numbers[0]]
    numbers=numbers[1:]
    for n in numbers:
        concatenated = np.concatenate((concatenated, x_train[n]), axis=1)
    return concatenated

def plot_numbers(x_train, numbers, columns=10, show_label=True, figsize=(20, 5)):
    plt.figure(figsize=figsize)
    numbers = np.array(numbers).reshape(-1, columns)
    concatenated = create_row(x_train, numbers[0])
    numbers = numbers[1:,:]
    for row in numbers:
        concatenated = np.concatenate((concatenated, create_row(x_train, row)))
    plt.imshow(concatenated, cmap='gray')
    plt.show()
 
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')