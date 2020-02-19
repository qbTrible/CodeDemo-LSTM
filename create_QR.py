# -*- coding: utf-8 -*- 
# @Time : 2019-12-13 11:23 
# @Author : Trible 

import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os


def randBgColor():
    return (random.randint(0, 180),
            random.randint(0, 180),
            random.randint(0, 180))


def randTextColor():
    return (random.randint(125, 255),
            random.randint(125, 255),
            random.randint(125, 255))


def randChar():
    return chr(random.randint(48, 57))


font_file = os.listdir(r"C:\Windows\Fonts")
font_name = []
bad_font = ["BSSYM7.TTF", "OUTLOOK.TTF", "REFSPCL.TTF", "MTEXTRA.TTF", "webdings.ttf", "WINGDNG2.TTF", "WINGDNG3.TTF",
            "wingding.ttf", "segmdl2.ttf", "marlett.ttf", "symbol.ttf", "LATINWD.TTF", "ITCBLKAD.TTF", "PALSCRI.TTF",
            "SNAP____.TTF", "GILSANUB.TTF", "KUNSTLER.TTF", "STCAIYUN.TTF", "FRSCRIPT.TTF", "PARCHM.TTF", "IndieFlower.ttf",
            "Gabriola.ttf", "JUICE___.TTF", "BOD_CR.TTF", "NIAGSOL.TTF", "NIAGENG.TTF", "GIGI.TTF", "VINERITC.TTF",
            "GLSNECB.TTF", "BOD_PSTC.TTF", "himalaya.ttf", "FREESCPT.TTF", "Candarali.ttf", "COLONNA.TTF", "PERBI___.TTF",
            "MSUIGHUR.TTF", "BOD_CI.TTF", "PoiretOne-Regular.ttf", "PLAYBILL.TTF", "msyi.ttf", "PER_____.TTF", "ROCC____.TTF",
            "ONYX.TTF", "BELL.TTF", "STKAITI.TTF", "FZSTK.TTF", "BRADHITC.TTF", "sylfaen.ttf", "HTOWERTI.TTF", "BELLI.TTF",
            "BRUSHSCI.TTF", "BRLNSB.TTF", "ERASLGHT.TTF", "PAPYRUS.TTF", "FORTE.TTF", "GILC____.TTF", "TCCM____.TTF",
            "PERI____.TTF", "couri.ttf", "GARA.TTF", "BASKVILL.TTF", "ITCEDSCR.TTF", "corbelli.ttf", "HATTEN.TTF"]
for file in font_file:
    if file.split(".")[-1] == "ttf" or file.split(".")[-1] == "TTF":
        if file not in bad_font:
            font_name.append(file)

w = 30 * 4
h = 60

for i in range(1000, 5000):
    image = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(w):
        for y in range(h):
            draw.point((x, y), fill=randBgColor())
    filename = []
    for t in range(4):
        ch = randChar()
        filename.append(ch)
        font_type = random.choice(font_name)
        # print(font_type)
        font = ImageFont.truetype(font_type, size=random.randint(32, 40))
        x_rand = random.randint(1, 10)
        y_rand = random.randint(1, 15)
        draw.text((30 * t + x_rand, y_rand), ch, fill=randTextColor(), font=font)
    image = image.filter(ImageFilter.BLUR)
    image_path = r"test_data"
    image.save("{0}/{1}.{2}.jpg".format(image_path, "".join(filename), i))
    print(i, filename)
