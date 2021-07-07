import sys
import argparse 
import cv2 as cv

#cd C:\Users\Admin\Desktop\summer\CV-SUMMER-CAMP-2021\practice\1_opencv
#cat_passport.py -i cat.jpg -m haarcascade_frontalcatface.xml

def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    img = cv.imread(input_image_path)
    
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Normalize image intensity
    norm_gray = cv.equalizeHist(gray)
    
    # Resize image
    small = cv.resize(img, (100, 75), interpolation=cv.INTER_CUBIC)

    # Detect cat faces using Haar Cascade
    detector = cv.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(75,75))
    print(rects)
                      
    # Draw bounding box
    for (i, (x,y,w,h))in enumerate(rects):
        cv.rectangle(img, (x,y), (x+w, y+w), (0,0,255),2)
        cv.putText(img, "Cat #".format(i+1), (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
        

    # Display result image
    cv.imshow("cat", img)
    cv.imshow("small", small)
    cv.waitKey(0)
    cv.imshow("gray", gray)
    cv.imshow("norm_gray", norm_gray)
    cv.waitKey(0)
    cv.destroyAllWindows();
    
    # Crop image
    x, y, w, h=rects[0]
    face = img[y:y+h,x:x+w]
        
    # Save result image to file
    for (i, (x,y,w,h))in enumerate(rects):
        cv.rectangle(img, (x,y), (x+w, y+w), (0,0,255),2)
        cv.imwrite('Cats_face'.format(i+1)+'.jpg', img[y:y+h,x:x+w])

    # Pasport
    x = 30
    y = 47
    norm_face = cv.resize(face, (202-x, 187-y), interpolation=cv.INTER_CUBIC)
    passport = cv.imread("pet_passport.png")
    cv.imshow("passport", passport)
    h,w,d = norm_face.shape
    for i in range(h):
        for j in range(w):
            for z in range(d):
                passport[i+y,j+x,z]=norm_face[i,j,z]
     
    cv.imshow("passport", passport)
    cv.waitKey(0)
    cv.destroyAllWindows();
    cv.imwrite("pet_passport.png", passport)
    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
