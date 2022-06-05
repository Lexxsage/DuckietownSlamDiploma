import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
from glob import glob

class EkfSlam():
    def __init__(self):
        self.velocities = np.loadtxt("interpolated_wheel_velocities.txt")
        self.time = np.loadtxt("interpolated_time.txt")
        self.range_bearing_dir = "range_bearing/"
        self.video_output_dir = 'video_output/'


    #wrap angles to pi
    def wraptopi(self, theta):
        if theta > np.pi :
            theta = theta  - 2*np.pi
        elif theta < -np.pi:
            theta = theta + 2*np.pi
        else:
            theta = theta

        return theta

    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    def create_map(self):
        T = 0.03

        # note that we will call the the estimates from the prediction step as x_check and P_check
        # whereas the corrected pose shall be called as x_hat and P_hat
        size = 3+28*2
        # initial conditions
        #we have 8 landmarks, pose <m_x,m_y>
        #state vector is 3 + 8*2, where 3 for the robot's pose and 8*2 for x and y coordinates of landmark
        x_0_hat = np.zeros((size,1))
        #init_landmark_cov =[1e1]*28*2
        init_landmark_cov =[10]*28*2
        P_0_hat = np.diag([0,0,0]+init_landmark_cov)
        cov_estimated_x = [0]
        cov_estimated_y = [0]
        cov_estimated_th = [0]
        #the initial cov for the landmarks is inf because we have no idea where they are


        #plotting variable
        plot_xlims = [1e6,-1e6]
        plot_ylims = [1e6,-1e6]
        plot_arrow_length = 1e-6
        ax = plt.figure().add_subplot(111)

        eigen_vals =[]
        ellipse_width =[]
        ellipse_height =[]
        ellipse_angle =[]
        nstd = 3
        ax = plt.subplot()
        cov = P_0_hat[0:2, 0:2]
        vals, vecs = eigsorted(cov)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        eigen_vals.append(vals)
        ellipse_width.append(w)
        ellipse_height.append(h)
        ellipse_angle.append(angle)

        # prediction step:
        sin = np.sin
        cos = np.cos
        v_var = 0.5#velocity variance
        om_var = 0.5#rotational velocity variance
        r_var = 1e-3 #(sensor) range variance
        b_var = 1e-3#(sensor) bearing variance
        w_k = np.diag([v_var,om_var]) #goes into motion model
        n_k = np.diag([r_var,b_var]) #goes into sensor model

        l = 0.1 #length from one wheel to the other
        #we will store our values here
        X_estimated  = [x_0_hat] #STATE VECTOR = where we think the robot is + the landmarks [x,y,theta, x1, y2, x2, y2, ....,]
        P_estimated = [P_0_hat] #COVARIANCE

        frame_count = 0
        previously_observed_landmarks =[]

        landmark_names = [24,96,25,86,7,23,65,30,57,87,80,85,32,78,10,31,79,9,11,61,0,1,2,3,4,5,12,8]

        d = 0.06 #distance between camera and axle


        for i in range(0, self.time.shape[0]):
            s = np.random.normal(0,0.001,size*2) #this is for process noise of the landmark
            s= s.reshape((size,2))

            #MOTION MODEL - PREDICTION STEP
            #differential drive model
            vel_left = self.velocities[i][0]
            vel_right = self.velocities[i][1]
            translational_velocity = (vel_left+vel_right)/2

            rotational_velocity = (1/l)*(vel_right-vel_left)

            theta_k_1 = self.wraptopi(X_estimated[-1][2]) #the latest theta
            P_hat = P_estimated[-1]
            F_K_1 = np.diag([1]*size)
            F_K_1[0][2] = -T*sin(theta_k_1)*translational_velocity
            F_K_1[1][2] = T*cos(theta_k_1)*translational_velocity

            w_jacobian_k = s.copy()
            w_jacobian_k[0][0] = T * cos(theta_k_1)
            w_jacobian_k[0][1] = 0
            w_jacobian_k[1][0] = T * sin(theta_k_1)
            w_jacobian_k[1][1] = 0
            w_jacobian_k[2][1] = T
            w_jacobian_k[2][0] = 0

            Q_prime_k = np.dot(w_jacobian_k, np.dot(w_k, w_jacobian_k.T)) #predicted linearized noise
            P_check_k = np.dot(F_K_1, np.dot(P_hat, F_K_1.T)) + Q_prime_k #predicted covariance

            X_check_k = X_estimated[-1].copy()
            #Note: only the position of the robot is predicted, not the landmarks - they stay where
            #they are (everything in world coordinates)
            X_check_k[0] = X_estimated[-1][0] + T * cos(theta_k_1) * translational_velocity #x
            X_check_k[1] = X_estimated[-1][1] + T * sin(theta_k_1) * translational_velocity #y
            X_check_k[2] = wraptopi(theta_k_1 + T*rotational_velocity) #theta

            #for a in range(3,len(X_check_k)):
                #X_check_k[a] = X_estimated[-1][a] + s[a][0] + s[a][1]


            #plot robot trajectory
            x_bot = [blah[0][0] for blah in X_estimated]
            y_bot = [blah[1][0] for blah in X_estimated]
            theta_bot = [blah[2][0] for blah in X_estimated]

            cov = P_check_k[0:2, 0:2]
            vals, vecs = eigsorted(cov)
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = Ellipse(xy=(X_estimated[-1][0], X_estimated[-1][1]),
                          width=w, height=h,
                          angle=angle, color='red')


            ell.set_facecolor('none')
            ax.add_artist(ell)

            ax.plot(x_bot,y_bot,'-k',linewidth=1,label='X_estimated')
            x_arrow = [x_bot[-1],x_bot[-1] + plot_arrow_length*np.cos(theta_bot[-1])]
            y_arrow = [y_bot[-1],y_bot[-1] + plot_arrow_length*np.sin(theta_bot[-1])]
            ax.plot(x_arrow,y_arrow,'-b',linewidth=0.5)

            #CORRECTION STEP correction step only occurs if there is an observation
            if int(self.time[i][0]) == 1: #there is an observation


                landmarks = np.loadtxt(self.range_bearing_dir + "range_bearing_frame" + format (frame_count,'06') +".txt")
                if landmarks.shape == (3,): #only 1 landmark
                    landmarks = landmarks.reshape(1,3) #reshape from shapeless to a shapely figure #o-la-la
                for j in range(landmarks.shape[0]):# iterate through the landmark list

                    if int(landmarks[j][0]) in landmark_names: #if the tag is a legit tag

                        landmark_index = landmark_names.index(int(landmarks[j][0])) #landmark's index in our names list
                        landmark_index = landmark_index *2 + 3 #landmark's index in the state vector

                        th_check_k = self.wraptopi(X_check_k[2,0])

                        if int(landmarks[j][0]) in previously_observed_landmarks: # we have seen this landmark before
                            #do the usual

                            #define these here for plotting
                            landmark_x_pose = X_check_k[landmark_index,0]
                            landmark_y_pose = X_check_k[landmark_index+1,0]
                            markersize=3 #indicates an existing landmark
                            markercolor='r'

                            #SEE NOTE 1

                            beta = X_check_k[landmark_index] - X_check_k[0,0] - d * cos(th_check_k)
                            gamma = X_check_k[landmark_index+1] - X_check_k[1,0]- d * sin(th_check_k)
                            alpha = np.sqrt(beta ** 2 + gamma ** 2)
                            range_landmark = alpha #estimated range
                            bearing_landmark = self.wraptopi(np.arctan2(gamma, beta) - th_check_k) #estimated bearing


                            measurement_range = float(landmarks[j][1]) #actual range
                            measurement_bearing = self.wraptopi(float(landmarks[j][2])) #actual bearing


                        else: #reintialize the start location of the landmark (before we had said it was at zero)
                            previously_observed_landmarks.append((int(landmarks[j][0]))) #now that we have seen it, put in in the observed list
                            measurement_range = float(landmarks[j][1]) #actual range
                            measurement_bearing = self.wraptopi(float(landmarks[j][2])) #actual bearing

                            #We don't have previous measurements to get the estimates from,
                            #so they are just equal to the first landmark.
                            range_landmark = 1*measurement_range
                            bearing_landmark = 1*measurement_bearing

                            #Because estimate = observed, innovation = 0. So have to manually set
                            #the position in the state vector
                            landmark_x_pose = X_check_k[0,0] + d*cos(th_check_k) + measurement_range*cos(self.wraptopi(self.wraptopi(measurement_bearing) + th_check_k))
                            landmark_y_pose = X_check_k[1,0] + d*sin(th_check_k) + measurement_range*sin(self.wraptopi(self.wraptopi(measurement_bearing) + th_check_k))

                            #Two cases, depending on whether this is the first landmark
                            #for that frame  or not (X_hat_k is only defined after first)

                            if j == 0:
                                X_check_k[landmark_index,0] = landmark_x_pose
                                X_check_k[landmark_index+1,0] = landmark_y_pose
                            else:
                                X_hat_k[landmark_index,0] = landmark_x_pose
                                X_hat_k[landmark_index+1,0] = landmark_y_pose


                            beta = landmark_x_pose - X_check_k[0,0] - d * cos(th_check_k)
                            gamma = landmark_y_pose - X_check_k[1,0]- d * sin(th_check_k)
                            alpha = np.sqrt(beta ** 2 + gamma ** 2)
                            # range_landmark = alpha #estimated range

                            #plotting
                            markersize=8 #indicates a new landmark
                            markercolor='g'


                        ax.plot(landmark_x_pose,landmark_y_pose,'x'+markercolor,markersize=markersize)
                        ax.annotate('{:1.0f}'.format(landmarks[j][0]),(landmark_x_pose,landmark_y_pose))

                        #jacobian of the sensor model wrt state vector
                        #(in the previous assignment, the landmark locations were known. Here they aren't, so jacobian is 2x5)
                        G_k_landmark = np.array([[-beta/alpha, -gamma/alpha, (beta*d*sin(th_check_k) - gamma*d*cos(th_check_k))/alpha,beta/alpha,gamma/alpha], \
                                   [gamma/alpha**2, -beta/alpha**2, (-d*sin(th_check_k)*gamma - d*cos(th_check_k)*beta)/alpha**2 -1, -gamma/alpha**2, beta/alpha**2]])

                        g_k_landmark = np.array([[range_landmark], [bearing_landmark]]) #predicted range and bear
                        measurements_landmark = np.array([[measurement_range],[measurement_bearing]])

                        #must create the A matrix that makes G_k into the full state vector shape
                        G_k_landmark = G_k_landmark.reshape(2,5)
                        A = np.zeros((5,size))
                        A[0][0] = 1
                        A[1][1] = 1
                        A[2][2] = 1
                        A[3][landmark_index] = 1
                        A[4][landmark_index + 1] = 1

                        #POTENTIAL ERROR SPOT: A should be matrix 15 on page 314 of textbook.

                        G_k = np.dot(G_k_landmark,A)

                        # we must create R'k
                        n_jacobian_k = np.array([[1, 0], [0, 1]])
                        R_prime_k_per_landmark = np.dot(n_jacobian_k, np.dot(n_k, n_jacobian_k.T))

                        # the kalman gain:
                        k_inside_brackets = np.linalg.inv(np.dot(G_k, np.dot(P_check_k, G_k.T)) + R_prime_k_per_landmark)
                        K_k = np.dot(np.dot(P_check_k, G_k.T), k_inside_brackets)

                        # corrector step:
                        z = np.dot(K_k, G_k)
                        identity = np.eye(z.shape[0])
                        P_check_k = np.dot(identity - z, P_check_k)


                        g_k_landmark = g_k_landmark.reshape(2,1)

                        innovation = measurements_landmark - g_k_landmark #actual measurements - estimated measurements

                        if j ==0:

                            X_CHECK_K = X_check_k

                            X_hat_k = X_CHECK_K + np.dot(K_k, innovation)
                        else:
                            X_CHECK_K = X_hat_k

                            X_hat_k = X_CHECK_K + np.dot(K_k, innovation)

                    else:
                        continue


                X_estimated.append(X_hat_k)
                P_estimated.append(P_check_k)


                cov_estimated = np.diag(P_check_k)
                cov_estimated_x.append(cov_estimated[0])
                cov_estimated_y.append(cov_estimated[1])
                cov_estimated_th.append(cov_estimated[2])

                cov = P_check_k[0:2, 0:2]
                vals, vecs = eigsorted(cov)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)
                eigen_vals.append(vals)
                ellipse_width.append(w)
                ellipse_height.append(h)
                ellipse_angle.append(angle)

                frame_count = frame_count + 1

                print(cov)
                print("******")
                #if frame_count == 1160:
                    #break


            else : #no observations
                X_estimated.append(X_check_k)
                P_estimated.append(P_check_k)


                cov_estimated = np.diag(P_check_k)
                cov_estimated_x.append(cov_estimated[0])
                cov_estimated_y.append(cov_estimated[1])
                cov_estimated_th.append(cov_estimated[2])

                cov = P_check_k[0:2, 0:2]
                vals, vecs = eigsorted(cov)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)
                eigen_vals.append(vals)
                ellipse_width.append(w)
                ellipse_height.append(h)
                ellipse_angle.append(angle)

            ###finish plotting

            #plot the current guesses for all of the landmarks which haven't been observed
            for landmark_index,lm in enumerate(landmark_names):
                if (len(landmarks)==0 or lm not in landmarks[:,0].astype(int)) and lm in previously_observed_landmarks:
                    #landmark was not observed on this frame and has
                    #not been plotted. so plot it.
                    landmark_index = landmark_index *2 + 3 #landmark's index in the state vector
                    landmark_x_pose = X_check_k[landmark_index]
                    landmark_y_pose = X_check_k[landmark_index+1]
                    ax.plot(landmark_x_pose,landmark_y_pose,'x',color='gray',markersize=3)
                    ax.annotate('{:1.0f}'.format(lm),(landmark_x_pose,landmark_y_pose),color='gray')

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            plot_xlims = [np.min([xlim[0],plot_xlims[0]]),np.max([xlim[1],plot_xlims[1]])]
            plot_ylims = [np.min([ylim[0],plot_ylims[0]]),np.max([ylim[1],plot_ylims[1]])]
            ax.set_xlim(plot_xlims)
            ax.set_ylim(plot_ylims)
            plot_arrow_length=0.05*np.sqrt((plot_xlims[1]-plot_xlims[0])**2+(plot_ylims[1]-plot_ylims[0]))
            ax.set_title('frame {:d}'.format(i))
            plt.savefig(self.video_output_dir+'{0:05d}.png'.format(i))
            ax.clear()