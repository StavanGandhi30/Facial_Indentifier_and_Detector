import face_recognition
import os
from PIL import Image, ImageDraw

def compare(oldImg,newImg):
    image_of_bill = face_recognition.load_image_file(oldImg)
    bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
    unknown_image = face_recognition.load_image_file(newImg)
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    compareResult = face_recognition.compare_faces([bill_face_encoding], unknown_face_encoding)
    return compareResult ##returns boolean; same=True, different=False

def num_of_people(Img):
    image = face_recognition.load_image_file(Img)
    face_locations = face_recognition.face_locations(image)
    return len(face_locations) ##returns int; number of people in image

def pull_faces(Img):
    image = face_recognition.load_image_file(Img)
    face_locations = face_recognition.face_locations(image)

    list_of_faces = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        list_of_faces.append(pil_image)
        # pil_image.show()
        # ## OR
        # pil_image.save(f'{top}.jpg')
    return list_of_faces ##returns list; array of faces data

def identify(imgPath, imgName, compareImgPath):
    ##All parameters Must be String.
    try:
        for compareImg in compareImgPath:
            knownImgEncoding = []
            knownImgName = imgName
            for value in imgPath:
                imgEncoding = face_recognition.load_image_file(value)
                imgEncoding = face_recognition.face_encodings(imgEncoding)[0]
                knownImgEncoding.append(imgEncoding)

            test_image = face_recognition.load_image_file(compareImg)
            face_locations = face_recognition.face_locations(test_image)
            face_encodings = face_recognition.face_encodings(test_image, face_locations)

            pil_image = Image.fromarray(test_image)
            draw = ImageDraw.Draw(pil_image)

            for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(knownImgEncoding, face_encoding)
                name = "Unknown Person"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = knownImgName[first_match_index]
                draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
                text_width, text_height = draw.textsize(name)
                draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
                draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))
            del draw
            pil_image.show()
            pil_image.save('identify.jpg')
    except Exception as error:
        print(error)
