import sys, math
import numpy as np
import random

# initial setting
h_c = 2*34/30.0*0.9  # 2.04 m
g   = 10.0
Tc  = math.sqrt(h_c/g)
StepTime2   = 0.4
target_vel  = 0.6
coeff_fp_LIP = np.array([math.cosh(StepTime2 / Tc) / math.sinh(StepTime2 / Tc) * Tc, -1 / math.sinh(StepTime2 / Tc) * Tc, 0]).reshape((3, 1))
coeff_vel_LIP =  np.array([math.sinh(StepTime2 / Tc) / Tc, math.cosh(StepTime2 / Tc), 0]).reshape((3,1))

#print coeff_vel_LIP

# index of the start of online estimation after several steps
n_oe_start  = 6
# index of the start of cyclic gait
c       = 2
# number of dataset for foot placement data store
n_fp    = 6

# setting for foot placement online estimation
walking_state_stack = np.zeros((n_fp + 1, 3))
dxcom_s = np.zeros((n_fp, 3))
fp_wrt_com  = np.zeros((n_fp, 1))

#number of dataset for velcoity estimation data store
n_vel   = 2

# setting for velocity online estimation
com_state_stack  = np.zeros((n_vel+1, 3))
com_state_s  = np.zeros((n_vel, 3))
vel_f_predict_s = np.zeros((n_vel, 1))


# fun_xcom: global com position
# fun_dxcom: global velocity
# fun_com_offset: com_offset, usually set as 1

def OnlineEsitmation(fun_xcom, fun_dxcom, fun_com_offset, fun_input_fp_wrt_com, stepindex):

    #Place the estimation of swing foot location, then this become the global support foot location
    fp_global = fun_xcom + fun_input_fp_wrt_com     # global foot placement point

    # use online estimation to obtain the model coefficients for veolcity estimation, continuous phase
    coeff_vel_OE = OnlineEsitmation_vel(fun_xcom - fp_global, fun_dxcom, fun_com_offset, stepindex)

    # use online estimation to obtain the model coefficients for foot placement estimation, discrete transitions
    coeff_fp_OE = OnlineEsitmation_fp(fun_xcom - fp_global, fun_dxcom, fun_com_offset, fun_input_fp_wrt_com, stepindex)

    # obtain the current walking state for velocity estimation
    walking_state_current = np.array([(fun_xcom - fp_global), fun_dxcom,1]).reshape(1,3)
    #print walking_state_current

    if stepindex <= (n_oe_start + c ):
        # Estimation of final velocity (LIPM)  continuous phase
        vel_f_predict = np.dot(walking_state_current, coeff_vel_LIP)[0][0]
        #print vel_f_predict

        walking_state_for_predict = np.array([vel_f_predict, target_vel, 1]).reshape(1, 3)

        # next placement of swing foot(LIPM)  discrete transitions
        fun_return_fp_wrt_com = np.dot(walking_state_for_predict, coeff_fp_LIP)[0][0]
        #print fun_return_fp_wrt_com

    else:
        # Estimation of final velocity (OE)  continuous phase
        vel_f_predict = np.dot(walking_state_current, coeff_vel_OE)[0][0]
        #print vel_f_predict
        # next placement of swing foot(OE)  discrete transitions
        walking_state_for_predict = np.array([vel_f_predict, target_vel, 1]).reshape(1, 3)
        fun_return_fp_wrt_com = np.dot(walking_state_for_predict, coeff_fp_OE)[0][0]
        #print fun_return_fp_wrt_com

    return fun_return_fp_wrt_com



def OnlineEsitmation_fp(fun_xcom, fun_dxcom, fun_com_offset, fp, stepindex):

    #data collection, based on Fisrt in First Out
    if stepindex >= c:
        if stepindex < (n_fp + c ):
            index = stepindex - c
            walking_state_stack[index] = [fun_xcom, fun_dxcom, fun_com_offset]
            if index >= 1:
                dxcom_s[index - 1,:] = [walking_state_stack[index - 1, 1], walking_state_stack[index, 1], fun_com_offset]

        else:

            for i in range(n_fp):
                walking_state_stack[i,:] =[walking_state_stack[i + 1, 0], walking_state_stack[i + 1, 1], fun_com_offset]

            walking_state_stack[n_fp,:] =[fun_xcom, fun_dxcom, fun_com_offset]

            for i in range(n_fp):
                dxcom_s[i,:] =[walking_state_stack[i, 1], walking_state_stack[i + 1, 1], fun_com_offset]


    if stepindex >= (c):

        if stepindex < (n_fp + c):
            #print(n_fp+c)
            # print(stepindex)
            # print("fp",fp)
            index = stepindex - c
            fp_wrt_com[index,:] =fp
        else:
            #print(stepindex)
            for i in range(n_fp - 1):
                fp_wrt_com[i,:] =fp_wrt_com[i + 1, 0]

            fp_wrt_com[n_fp-1,:] =fp



    #LIPM model coff for foot placement estimation
    coeff_fp_LIP = np.array([math.cosh(StepTime2 / Tc) /math.sinh(StepTime2 / Tc) * Tc, -1/math.sinh(StepTime2 / Tc) * Tc,0]).reshape(3,1)

    # Tikhonov regularization
    # https: // en.wikipedia.org / wiki / Tikhonov_regularization  # Generalized_Tikhonov_regularization
    # setting of optimation function

    # regression term
    # P will set higher weight to lastest step
    P = []
    for i in range(n_fp):
        P.append(i)

    # gain_P is used to weight the influence of the regression term
    gain_P = 0.1

    P = np.array(np.diag(P)) * gain_P

    # regularisation term
    Q = np.array([1, 1, 0.01])
    Q = np.diag(Q)

    # gain_Q is used to weight the influence of the regularisation term
    gain_Q = 0.1  # 0.5

    Q = np.array(Q) * gain_Q


    if stepindex < (n_oe_start + c ):
        fun_return_coeff_fp = coeff_fp_LIP
    else:

        # ----------------Tikhonov regularization - ----------------

        #fun_return_coeff_fp = coeff_fp_LIP + (np.dot(np.dot(np.transpose(dxcom_s),P),dxcom_s)+ Q) / np.dot(np.dot(np.transpose(dxcom_s), P), (fp_wrt_com - np.dot(dxcom_s,coeff_fp_LIP)))

        #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html
        #https://stackoverflow.com/questions/15887191/python-pseudo-inverse-and-determinant-of-a-vector
        #fun_return_coeff_fp = coeff_fp_LIP + np.linalg.pinv(np.transpose(dxcom_s) * P * dxcom_s + Q)*(np.transpose(dxcom_s) * P * (fp_wrt_com - np.dot(dxcom_s,coeff_fp_LIP)))

        fun_return_coeff_fp = coeff_fp_LIP + np.linalg.pinv(np.dot(np.dot(np.transpose(dxcom_s),P),dxcom_s)+ Q) * np.dot(np.dot(np.transpose(dxcom_s), P), (fp_wrt_com - np.dot(dxcom_s,coeff_fp_LIP)))



        # --------------- Least Squares - ----------------

        #fun_return_coeff_fp = pinv(transpose(dxcom_s) * P * dxcom_s) * transpose(dxcom_s) * P * fp_wrt_com

        #fun_return_coeff_fp = lsqlin(eye(3), coeff_fp_LIP, [], [], dxcom_s, fp_wrt_com)


    #debug section

    #print('--------------')
    # print("coeff_fp_LIP",coeff_fp_LIP)
    # # print(P)
    # # print(Q)
    # print("walking_state_stack",walking_state_stack)
    # print("dxcom_s",dxcom_s)
    # print("fp_wrt_com",fp_wrt_com)
    # print("fun_return_coeff_fp",fun_return_coeff_fp)

        # pinv(dxcom_s) * fp_wrt_com

    return fun_return_coeff_fp


def OnlineEsitmation_vel(fun_xcom, fun_dxcom, fun_com_offset, stepindex):

    # data collection, based on Fisrt in First Out
    if stepindex >= c:

        if stepindex < (n_vel + c ):

            index = stepindex - c
            com_state_stack[index,:] = [fun_xcom, fun_dxcom, fun_com_offset]

            if index >= 1:
                com_state_s[index - 1,:] = [com_state_stack[index - 1, 0], com_state_stack[index - 1, 1], fun_com_offset]
                vel_f_predict_s[index - 1,:] = com_state_stack[index, 1]


        else:

            for i in range(n_vel):
                com_state_stack[i,:]    =    [com_state_stack[i + 1, 0], com_state_stack[i + 1, 1], fun_com_offset]


            com_state_stack[n_vel,:]     =    [fun_xcom, fun_dxcom, fun_com_offset]

            for i in range(n_vel):
                com_state_s[i,:]    = [com_state_stack[i, 0], com_state_stack[i, 1], fun_com_offset]
                vel_f_predict_s[i,:]    =   com_state_stack[i + 1, 1]


    #LIPM model coff for velocity estimation
    coeff_vel_LIP =  np.array([math.sinh(StepTime2 / Tc) / Tc, math.cosh(StepTime2 / Tc), 0]).reshape(3,1)


    # setting of optimation function
    # regression term
    # P will set higher weight to lastest step
    P = []
    for i in range(n_vel):
        P.append(i)

    # gain_P is used to weight the influence of the regression term
    gain_P = 0.1

    P = np.array(np.diag(P)) * gain_P

    # regularisation term
    Q = np.array([1, 1, 0.01])
    Q = np.diag(Q)

    # gain_Q is used to weight the influence of the regularisation term
    gain_Q = 0.1  # 0.5

    Q = np.array(Q) * gain_Q


    if stepindex < (n_oe_start + c ):

        fun_return_coeff_vel = coeff_vel_LIP

    else:

        #Tikhonov regularization
        # https: // en.wikipedia.org / wiki / Tikhonov_regularization  # Generalized_Tikhonov_regularization
        fun_return_coeff_vel = coeff_vel_LIP + np.linalg.pinv(np.dot(np.dot(np.transpose(com_state_s),P), com_state_s) + Q) * np.dot(np.dot(np.transpose(com_state_s),P), (vel_f_predict_s - np.dot(com_state_s,coeff_vel_LIP)))

      #fun_return_coeff_vel = coeff_vel_LIP + (np.dot(np.dot(np.transpose(com_state_s),P), com_state_s) + Q) / np.dot(np.dot(np.transpose(com_state_s),P), (vel_f_predict_s - np.dot(com_state_s,coeff_vel_LIP)))



        #debug section

        #     disp('--------------')
        #     P
        #     Q
        #     com_state_stack
        #     com_state_s
        #     vel_f_predict_s
        #     fun_return_coeff_vel
        #     pinv(com_state_s) * vel_f_predict_s
        #     pause(0.1)


    return fun_return_coeff_vel

if __name__=="__main__":

    fun_xcom = random.uniform(1, 2)
    fun_dxcom = random.uniform(1, 2)
    fun_com_offset = 1
    fun_input_fp_wrt_com = random.uniform(1, 2)

    xcom =0.645 #[0.26666666666666666, 0.3970177968343096, 0.66945536931355765, 0.96175034840901663, 1.2576859792073565,
     #1.5636232693990069, 1.8778284390767412, 2.1833260854085283, 2.4753301938374834, 2.7599485715230303,
     #3.0358413060506182, 3.3235386212666826, 3.6824611028035479, 4.2124780019124346, 5.1279939015706377,
     #6.7211907704671221]
    dxcom = 0.594#[0.0, 0.69670289754867554, 0.7679283618927002, 0.78619599342346191, 0.79436671733856201,
            # 0.84038209915161133, 0.84453326463699341, 0.78709721565246582, 0.76361393928527832,
            # 0.74587303400039673, 0.71890383958816528, 0.85366612672805786, 1.0921984910964966,
            # 1.8219931125640869, 3.6072511672973633, 3.8810050487518311]
    input_fp_wrt_com = 0.109  #[-0.26666666666666666, 0.10887806613155354, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414, 0.11272678309649414]



    stepindex = 10

        # fp = OnlineEsitmation_fp(fun_xcom, fun_dxcom, fun_com_offset, fun_input_fp_wrt_com, stepindex)
        # print(fp)

        # vel = OnlineEsitmation_vel(fun_xcom, fun_dxcom, fun_com_offset, stepindex)
        # print(vel)

    out_fp_wrt_com = OnlineEsitmation(xcom, dxcom, fun_com_offset, input_fp_wrt_com, stepindex)

    print out_fp_wrt_com

