import matplotlib.pyplot as plt

def show_images(imgs, labels, col, row):
    show_imgs = imgs[0:col * row]
    plt.figure()
    for i in range(0, col * row):
        plt.subplot(row, col, i+1)
        plt.imshow(show_imgs[i], cmap="Greys")
        plt.xticks([])
        plt.yticks([])
        plt.title(labels[i])
        print(show_imgs[i])
    plt.show()