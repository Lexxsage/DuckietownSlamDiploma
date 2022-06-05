import json, codecs
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

class TagReader():
    def __init__(self):
        self.tags_dir = "TagDetections/"
        self.image_dir = "Undistored/"

    def read_tags(self):
        theta = (10/360) * 2*np.pi
        #The 15-degree tilt of our camera - rotation matrix
        rotation = np.array([[1,0,0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta),np.cos(theta)]])

        #get all json files
        histogram=[]
        all_json = sorted(glob(self.tags_dir+'*.json'))

        x_landmark =[]
        y_landmark = []
        x_r_landmark =[]
        y_r_landmark =[]
        ranges_to_plot = list(range(500,700,5))
        for file_path in all_json:
            #see https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
            file_text = codecs.open(file_path, 'r', encoding='utf-8').read()
            file_data = json.loads(file_text)
            histogram.append(len(file_data))

            file_id = file_path[-11:-5]
            image_path = self.image_dir+'undistorted_frame'+file_id+'.png'

            file_index = int(file_id)

            if len(file_data)>0:
                #Convert all of the lists back to numpy arrays
                x_landmark = []
                y_landmark = []
                x_r_landmark = []
                y_r_landmark = []
                tag_ids = []
                for i in range(len(file_data)):
                    tag_dict = file_data[i]
                    for k in tag_dict:
                        if isinstance(tag_dict[k],list):
                            tag_dict[k] = np.asarray(tag_dict[k])
                    file_data[i] = tag_dict
                    print(file_data[i]['pose_t'])
                    pose = np.dot(rotation, file_data[i]['pose_t'])
                    rangesy = np.sqrt(pose[2]**2 + -pose[0]**2)
                    bearing = np.arctan2((-pose[0]),pose[2])

                    if file_index in ranges_to_plot:
                        pose = np.dot(rotation,file_data[i]['pose_t'])

                        x_landmark.append(file_data[i]['pose_t'][2])
                        y_landmark.append(-file_data[i]['pose_t'][0])
                        x_r_landmark.append(pose[2])
                        y_r_landmark.append(-pose[0])
                        tag_ids.append(file_data[i]['tag_id'])
                if len(x_landmark) > 0:
                    print(file_path)

                    fig_plot = plt.figure(figsize=(10,5))
                    ax_plot = fig_plot.add_subplot(121)
                    ax_imag = fig_plot.add_subplot(122)

                    #plot image
                    img = mpimg.imread(image_path)
                    ax_imag.imshow(img)
                    ax_imag.set_title(image_path.split('/')[-1])

                    for ii in range(len(tag_ids)):
                        ax_plot.plot(x_landmark[ii], y_landmark[ii], 'ob')
                        ax_plot.plot(x_r_landmark[ii], y_r_landmark[ii], 'ok')
                        ax_plot.text(x_r_landmark[ii], y_r_landmark[ii], '   '+str(tag_ids[ii]) )
                    ax_plot.plot(0, 0, 'or')
                    ax_plot.set_title(file_path.split('/')[-1]+'\n'+'# tags = '+str(len(x_landmark)))
                    ax_plot.set_xlim((-0.05,3))
                    ax_plot.set_ylim((-3,3))
                    plt.show()

            else:
                print('No tags detected')