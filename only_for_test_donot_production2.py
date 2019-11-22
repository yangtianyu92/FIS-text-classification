#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 14:32

import os
from check_id import checkIdcard

#GPUID = '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import cv2
import json
import time
import uuid
import base64
from PIL import Image
import model
from config import DETECTANGLE
from apphelper.image import xy_rotate_box,box_rotate,solve
from application import trainTicket
import idcard2
import timeout_decorator
import numpy as np
import tornado.web
import tornado.ioloop
from tornado.options import define, options, parse_command_line
import json
define('port', default=8888, help='run on the port', type=int)




def pre_recognize(img):

    _,result,angle= model.model(img,detectAngle=True,
                                config=dict(MAX_HORIZONTAL_GAP=50,MIN_V_OVERLAPS=0.6,MIN_SIZE_SIM=0.6,TEXT_PROPOSALS_MIN_SCORE=0.1,TEXT_PROPOSALS_NMS_THRESH=0.3,TEXT_LINE_NMS_THRESH = 0.7),
                                leftAdjust=True,
                                rightAdjust=True,
                                alph=0.01)
    return result


@timeout_decorator.timeout(3)
def image_recognize(img):
    return pre_recognize(img)



class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.write("You should not use [GET] method, please use [POST] method, an json include ['ImgBase64'] key and it contain base64 encode binary picture file")


    def post(self):
        data = json.loads(self.request.body.decode("utf-8"))
        frontImgStringBase64 = data['ImgBase64']
        frontImgBinary = base64.b64decode(frontImgStringBase64)
        
        pic_name = uuid.uuid1().__str__()+".jpg"
        b_pic_name = "conv_" + pic_name
        path = './cardPicture/{}'.format(pic_name)
        path2 = './cardPicture/{}'.format(b_pic_name)
        with open(path,'wb') as f:
            f.write(frontImgBinary)
        os.system("ffmpeg -i " + path + " " + path2)

        img = cv2.imread(path2) ##GBR
        b, g, r = cv2.split(img)
        card = cv2.merge([r,g,b])

        time_begin = time.time()

        text_recognize = image_recognize(card)
        res_id_card = idcard2.idcard(text_recognize)
        res_card_info_json = res_id_card.res

        timeTake = time.time() - time_begin
        self.write(json.dumps({"res":res_card_info_json, "status":"", "timeuse":str(timeTake)}, ensure_ascii=False))
        print("time elapsed....."+str(timeTake))
        os.remove(path)
        os.remove(path2)


def main():
    # parse_command_line()
    app = tornado.web.Application([(r'/ocr', MainHandler)])
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    img = cv2.imread('test1.jpg')
    try:
        pre_recognize(img)
    except:
        pass

    print("early start OK!")
    main()

    
