import base64

def base64_to_image(base64_string):
    output_path = 'img.png'
    image_data = base64.b64decode(base64_string)
    with open(output_path, 'wb') as output_file:
        output_file.write(image_data)
    return output_path
