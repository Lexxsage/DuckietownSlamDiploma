import numpy as np
from scipy.interpolate import interp1d
import pylab as plt

def Interpolator():
    wheel_file_path = ''
    image_file_path = ''
    out_path = ''

    def safe_file_paths():
        global wheel_file_path
        global image_file_path
        global out_path
        wheel_file_path = str(input())
        image_file_path = str(input())
        out_path = str(input())


    def interpolate():
        global wheel_file_path
        global image_file_path
        global out_path

        wheel_data = np.loadtxt(wheel_file_path)
        image_time = np.loadtxt(image_file_path)

        l_vel = wheel_data[:,0]
        r_vel = wheel_data[:,1]
        wheel_time = wheel_data[:,2]

        #convert to seconds
        image_time = image_time/1e9
        wheel_time = wheel_time/1e9

        #median time between images
        median_image_time = np.median(np.diff(image_time))

        #find large gaps in the image_time and fill them with wheel time
        gapless_image_time = []
        has_image = []
        max_interval = 0.2
        for i in range(len(image_time)-1):
            #add the image time, which we know has an associated image
            gapless_image_time.append(image_time[i])
            has_image.append(True)

            #check if there's a gap between this image time point and the
            #next image time point. If this is too big (>max_interval), then we
            #will fill it with times from the wheel velocity data
            if image_time[i+1]-image_time[i]>max_interval:
                wheel_time_to_add = np.arange(image_time[i],image_time[i+1],median_image_time)
                has_image += [False]*len(wheel_time_to_add)
                gapless_image_time += wheel_time_to_add.tolist()

        gapless_image_time=np.array(gapless_image_time)
        has_image = np.array(has_image)

        #interpolation functions
        l_interp_func = interp1d(x=wheel_time,y=l_vel,kind='linear',bounds_error=False,fill_value=0)
        r_interp_func = interp1d(x=wheel_time,y=r_vel,kind='linear',bounds_error=False,fill_value=0)

        l_vel_interp = l_interp_func(gapless_image_time)
        r_vel_interp = r_interp_func(gapless_image_time)

        #write  out
        np.savetxt(out_path+'interpolated_time.txt',np.transpose((has_image.astype(int),gapless_image_time)),
            header='HAS_IMAGE (0=false,1=true), TIME')
        np.savetxt(out_path+'interpolated_wheel_velocities.txt',np.transpose((l_vel_interp,r_vel_interp)),
                   header='Left velocity, right velocity')
