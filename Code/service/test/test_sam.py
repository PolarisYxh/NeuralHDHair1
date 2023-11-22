try:
    from http_interface import *
except:
    from ..http_interface import *
    
img = cv2.imread("/home/yangxinhang/NeuralHDHair/female_9.jpg")
img = cv2.resize(img,(640,640))
segall = faceParsingInterface(os.path.dirname(__file__))
imgB64 = cvmat2base64(img)#framesForHair[0] 640,640

parts,masks = segall.request_faceParsing("", 'img', imgB64)

segall = segmentAllInterface(os.path.dirname(__file__))
imgB64 = cvmat2base64(img)#framesForHair[0] 640,640

masks = segall.request_faceParsing("", 'img', imgB64,np.array([[320,320],[320,100]]),[1,0])
parsing = np.zeros_like(img[:,:,0])
parsing[masks] = 255
save_parsing = np.zeros_like(img)
save_parsing[masks] = img[masks]
cv2.circle(save_parsing, tuple([320,200]), 1, (255, 0, 0), 2)
cv2.imwrite("1.png",save_parsing)