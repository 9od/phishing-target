import time
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import tkinter.font as tkFont

def phishResult():
    messagebox.showwarning('Wraning', 'phishing threat')


def showImg():
    imgpath = './imgs/bound.jpg'
    root1 = tk.Toplevel()
    root1.geometry('1400x750')
    root1.title('PhishWeb')
    Img = Image.open(imgpath)
    Img = Img.resize((1245, 520))
    img_ = ImageTk.PhotoImage(Img)

    ft2 = tkFont.Font(size=15, weight=tkFont.BOLD)
    tk.Label(root1, compound='center', image=img_).place(x=0, y=30)
    tk.Label(root1, text="WARNING!: This is a phishing website for Microsoft, "
                         "its domain modifies microsoft to macrosoft",
             fg="red", bg="white", font=ft2).place(x=30, y=10, width=1000, height=50)
    tk.Button(root1, text='Quit', width=10, command=root1.destroy) \
        .place(x=1000, y=600, width=80, height=40)
    root1.mainloop()


if __name__ == '__main__':
    root = tk.Tk()

    root.geometry('1000x600')
    root.title('PhishMail')

    imgpath1 = './imgs/m2.png'
    Img1 = Image.open(imgpath1)
    Img1 = Img1.resize((800, 534))
    img1 = ImageTk.PhotoImage(Img1)
    imgpath2 = './imgs/m2_.png'
    Img2 = Image.open(imgpath2)
    Img2 = Img2.resize((800, 534))
    img2 = ImageTk.PhotoImage(Img2)
    tk.Label(root, compound='center', image=img1).place(x=100, y=20)

    for t in range(10):
        canvas = tk.Canvas(width=50, height=50, bg='white')
        canvas.place(x=440, y=400)
        imgpro = './imgs/load' + str(t % 4 + 1) + '.png'
        Imgpro = tk.PhotoImage(file=imgpro)
        canvas.create_image(27, 27, image=Imgpro, tag="pic")
        canvas.update()
        time.sleep(0.2)
        canvas.delete("pic")
        if t >= 2:
            ft2 = tkFont.Font(size=13, weight=tkFont.BOLD)
            tk.Label(root, compound='center', image=img2).place(x=100, y=20)
            tk.Label(root, text="Scanning...",fg="red", bg="yellow", font=ft2) \
                     .place(x=400, y=130, width=100, height=30)
        if t == 9:
            phishResult()
            showImg()

    root.mainloop()
