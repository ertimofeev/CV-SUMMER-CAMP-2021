import sys
import argparse 
import cv2 as cv

#cd C:\Users\Admin\Desktop\summer\CV-SUMMER-CAMP-2021\practice\1_opencv
#cat_passport -i cat.jpg -m haarcascade_frontalcatface.xml
def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    img = cv.imread(input_image_path)
    #print(img)
    # Convert image to grayscale

    # Normalize image intensity

    # Resize image

    # Detect cat faces using Haar Cascade
    detector = cv.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(75,75))
    print(rects)
    # Draw bounding box

    # Display result image
    cv.imshow("cat", img)
    cv.waitKey(0)
    cv.ddestroyAllWindows();
    # Crop image

    # Save result image to file

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
