from google.cloud import vision
from google.oauth2 import service_account



# Set up the Vision client with the service account key file
vision_client = vision.ImageAnnotatorClient(
    credentials=service_account.Credentials.from_service_account_file('path/to/keyfile.json'))

# Define the folder path
folder_path = '/Users/pankajrathi/job/01_Mietverträge'

# Loop through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image file
        file_path = folder_path + filename
        with io.open(file_path, 'rb') as image_file:
            content = image_file.read()

        # Detect text in the image
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations

        # Print the text
        if texts:
            print('Text in image {}:'.format(filename))
            for text in texts:
                print(text.description)
        else:
            print('No text detected in image {}'.format(filename))