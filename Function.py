import numpy as np
import cv2
import matplotlib.pyplot as plt 


#%%
def grayscale_values(image, y_position):
    # Linienbild erstellen
    #line_image = create_horizontal_line(image, y_position)

    # x-Achse und Grauwerte initialisieren
    x_values = np.arange(image.shape[1])
    gray_values = []

    # Grauwerte entlang der Linie sammeln
    for x in x_values:
        gray_values.append(image[y_position, x])

    return gray_values, x_values

#%%
# Plot erstellen
def plot_values (gray_values, x_values):
    plt.plot(x_values, gray_values)
    plt.title('Grauwerte entlang der Suchlinie')
    plt.xlabel('x-Achse')
    plt.ylabel('Grauwert')
    plt.show()

#%%
def white_or_black (image, color):
     
    if color == "w":
        img_out = image
    else:
        if color == "s":
            img_out = 255 - image
        else:
            raise ValueError("Ungültige Farbauswahl. Gültige Optionen sind 's' oder 'l'.")

    return img_out

#%%
def find_thresh (image, persentage):

    h = image.shape[1]
    all_grayscale_values = []
    #x_values = []

    for y in range(int(h/2), int(h-1), int(h/8)):
        gray_values, x_values = grayscale_values(image, y)
        all_grayscale_values.extend(gray_values)

    max_gray = max(all_grayscale_values)
    
    sorted_values = sorted(all_grayscale_values)  # Sortiere die Grauwerte aufsteigend
    num_values = len(sorted_values)  # Anzahl der Grauwerte

    thresh_percent = 1-persentage
    
    index_1 = int(num_values * thresh_percent)  # Index für den 10% Punkt
    threshold_2 = int(max_gray - (max_gray*persentage))

    print(index_1)
   
    threshold_1 = sorted_values[index_1]

    return threshold_1, threshold_2

#%%

#%%
def find_largest_component (image):
        '''
        find all connected Components, search for the largest
        Output: Image with the largest Component
        '''
        # Finde alle zusammenhängenden Elemente
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Finde das größte zusammenhängende Element
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Erstelle ein Bild, das nur das größte zusammenhängende Element enthält
        largest_component_image = np.zeros_like(image)
        largest_component_image[labels == largest_component_label] = 255

        return largest_component_image

#%%
def seg_largest_component(image):
    '''
    finds the orientation lines and segment them
    Outout: Image withe segmentated lines
    '''
    img_dilate = cv2.dilate(image.astype('uint8'), np.ones((3,3)), iterations=7)
    #plt.imshow(img_dilate)

    img_largest = find_largest_component(img_dilate)
    #plt.imshow(img_largest)

    #Um evtl. falsch verbundene Komponenten zu entfernen nocheinmal erode, find_largest_component, dilate
    iterations_2 = 8

    img_erode = cv2.erode(img_largest.astype('uint8'), np.ones((3,3)), iterations=iterations_2)
    #plt.imshow(img_erode)

    img_largest_2 = find_largest_component(img_erode)
    #plt.imshow(img_largest_2)

    img_out = cv2.dilate(img_largest_2.astype('uint8'), np.ones((3,3)), iterations= iterations_2*2)

    return img_out

#%%
def seg_orientation_lines(image, color):
    '''
    segmentiert aus einem Imput Bild die Ortientierungslinien
    Output: bw_image with contours of orientation lines
    '''
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.gray()

    #Gauss für bessere Ergebnisse bei finde thresh
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    #color = "w"   #Farbe der Orientierungslinie wählen: w für weiß und s für schwarz
    img_b_or_w = white_or_black(img_blur, color)

    thresh_1, thresh_2 = find_thresh(img_b_or_w, 0.13)  # oberen möglichen 13% ab maximalem Grauwert

    print("1:", thresh_1)
    print("2:", thresh_2)

    t, seg = cv2.threshold(img_b_or_w,thresh_2,255,cv2.THRESH_BINARY)

    #segmentiert das größe zusammenhängende Objekt 
    img_out = seg_largest_component(seg)

    return img_out

#%%