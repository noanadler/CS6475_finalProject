# plot of log line
from tkinter import *
from tkFileDialog import askopenfilename
import prepare_db
import cv2
import flower_identifier

def run_flower_identifier(input_file, label2):
    global result
    if input_file == "":
        label2.config(text="Please Select Input File")
        label2.pack()
        return input_file
    test_img_dic = {}
    results = {}
    test_img = cv2.imread(input_file)
    test_img_dic["test"] = test_img
    test_img_dic = prepare_db.remove_background(test_img_dic)
    test_img_dic = prepare_db.rezise_images(test_img_dic)
    test_img = test_img_dic["test"]

    image_dir = "db/same_size_images"
    # images is a dictionary with flower name as key and flower images as value
    images = flower_identifier.readImages(image_dir)

    for key in images.keys():
        results[key] = 0

    results[flower_identifier.find_sift_best_match(images, test_img)] += 1
    results[flower_identifier.find_color_histogram_best_match(images, test_img)] += 1
    results[flower_identifier.find_gradients_histogram_best_match(images, test_img)] += 1
    result =  max(results, key=results.get)
    result = result.upper()
    label2.config(text="The Flower is: " + result, width=40, font='size, 30')
    label2.pack(side='top', padx=20)

    return result

def get_input_file_name(label1, label2, width, height):
    global input_file
    input_file= askopenfilename()
    displayImage = PhotoImage(file = input_file)
    label1.configure(image = displayImage, width=width, height=height)
    label1.image = displayImage
    label1.pack()
    label2.config(text="")
    button2 = Button(root, text='Identify Flower', font='size, 30', width=20, command= lambda: run_flower_identifier(input_file, label2))
    button2.grid(row=0, column=0)
    button2.pack(side='top', padx=20)


# define root window
root = Tk()
root.title("WhatsThatFlower")
# create frame to put control buttons onto
frame = Frame(root, bg='grey', width=200, height=200)
frame.pack(fill='x')
# set label properties
width = 600
height = 400

label1 = Label(root, width=width, height=height)
label1.pack()
label2 = Label(root, width=width, height=height)

button1 = Button(frame, text='Choose Input File', font='size, 30', width=20, command= lambda: get_input_file_name(label1, label2, width, height))
button1.pack(side='top', padx=20)
root.mainloop()
