withoutMask = preds[0][0]
withMask = preds[0][1]
print("WITHOUT MASK") if withoutMask > withMask else print(
    "=======================WITH MASK===================================")


if withoutMask > withMask:
    img1 = face_recognition.load_image_file('yid5.jpg')
    img2 = face_recognition.load_image_file('kl.jpg')

    face_encoding1 = face_recognition.face_encodings(img1)[0]
    face_encoding2 = face_recognition.face_encodings(img2)[0]

    known_face_encoding = [
        face_encoding1,
        face_encoding2,
        # face_encoding3
    ]

    # unkownImage = face_recognition.load_image_file(faces)
    # print('FACES:  ', frame)
    face_encoding_unknown = face_recognition.face_encodings(frame)

    for unknown_face_encoding in face_encoding_unknown:

        res = face_recognition.compare_faces(
            known_face_encoding, unknown_face_encoding)

        name = "unknown"

        if res[0]:
            name = "Yididya Samuel"
        elif res[1]:
            name = "Kalkidan Samuel"
        # elif res[2]:
        #     name = "KAL2nd"

        print(f'Found {name} in the picture!')
# return a 2-tuple of the face locations and their corresponding
# locations

# else:
return (locs, preds)
