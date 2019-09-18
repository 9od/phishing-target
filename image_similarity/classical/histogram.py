from PIL import Image


def make_regalur_image(img, size=(256, 256)):
    return img.resize(size).convert('RGB')


def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)


def calc_similar(li, ri):
    return hist_similar(li.histogram(), ri.histogram())


if __name__ == '__main__':
    img1 = Image.open('1.jpg')
    img1 = make_regalur_image(img1)
    img2 = Image.open('2.jpg')
    img2 = make_regalur_image(img2)
    print(calc_similar(img1, img2))