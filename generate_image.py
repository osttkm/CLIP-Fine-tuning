import random
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

image_size = (224, 224)
color = np.array([(255,0,0),(128,128,0),(255,255,0),(255,0,255),(192,192,192),(0,255,255),(0,255,0),(0,128,0),(128,128,128),(0,0,255)])
color_name = np.array(['red','olive','yellow','fuchsia','silver','aqua','lime','green','gray','blue'])
circle_place = np.array(['upper left','left','lower left','upper','center','lower','upper right','right','lower right'])
back_ground_image = list(Path('./image/base_image/').glob('*.jpg'))

file = open("./image/detail.txt", "w")
file.close()



# 9分割の正方形のサイズと位置を計算
square_size = image_size[0] // 3
squares = []
for i in range(3):
    for j in range(3):
        left = i * square_size
        upper = j * square_size
        right = left + square_size
        lower = upper + square_size
        squares.append((left, upper, right, lower))

count=0
for circle_color in color:
    delete_color_name = color_name[(color==circle_color).sum(axis=1)==3][0]
    delete_index = np.where(color_name==delete_color_name)[0][0]
    back_ground_color = np.delete(color_name,delete_index)
    for b_image in back_ground_image:
        for place in circle_place:

            # ランダムに正方形を選択
            selected_square = random.choice(squares)

            # ランダムな半径を生成
            min_radius = square_size // 4
            max_radius = square_size // 2
            circle_radius = random.randint(min_radius, max_radius)

            # 丸を描画
            # import pdb;pdb.set_trace()
            # image = Image.new("RGB", image_size, tuple(color[color_name==back_ground_color][0]))
            # draw = ImageDraw.Draw(image)
            image = Image.open(str(b_image))
            image = image.resize((224,224))
            circle_center = (
                selected_square[0] + square_size // 2,
                selected_square[1] + square_size // 2
            )
            draw = ImageDraw.Draw(image)
            draw.ellipse((circle_center[0]-circle_radius, circle_center[1]-circle_radius,
                        circle_center[0]+circle_radius, circle_center[1]+circle_radius),
                        outline=tuple(circle_color), width=5)
            # import pdb;pdb.set_trace()
            caption = f'''A photo of a {color_name[(color==circle_color).sum(axis=1)==3][0]} circles drawn in the {place} portion of the {str(b_image.name).split('.')[0]} background'''
            image.save(f"./image/{count}.png")
            image.close()
            with open("./image/detail.txt", "a") as file:
                file.write(f"./image/{count}.png$${caption}\n")
            count+=1
        
