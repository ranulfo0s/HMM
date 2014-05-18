import pandas as pd
import array
from datetime import *
import numpy as np
from sklearn import linear_model
from sklearn.hmm import GaussianHMM
import matplotlib.pyplot as plt


#Just a function to compare the lists.
def comp(A, B):
    suma = 0
    for x in range(len(B)):
        if A[x] == B[x]:
            suma += 1
    return (suma/float(len(B)))



########################################################################################






########################################################################################
def get_features_target(Path):

    ##Read the .CSV 
    data = pd.read_csv(Path)

    #Rename the columns with this respctive names to easy control the file
    data.columns = ['itime','x','y','z','date','time','estimate','position']

    #Transform the columns time from string to datetime
    data['time'] = pd.to_datetime(data['time'])



    #Inicialize variables and lists    
    clips = 0

    clipsF = 0

    start_point = 0


    inicial_time = data['time'][start_point]

    end_point = len(data['time']) - 1

    X = []
    X.append([])

    Y = []
    Y.append([])

    Z = []
    Z.append([])

    clips_range = timedelta(seconds = 2)

    somed = data['time'][end_point] - inicial_time

    n_clips = int(round(((somed.total_seconds()/2)*10)/10))

    n_features = 6

    features = []

    state = []

    target = []

    walking, misc,lying, standing, wheeling, sitting = 0,0,0,0,0,0

    ########################################################################################
    for k in range(start_point,end_point):
            #0 -> Lying
            #1 -> Walking
            #2 -> Standing
            #3 -> Wheeling
            #4 -> Sitting
        if data['estimate'][k] == 'Walking':
            walking += 1
        elif data['estimate'][k] == 'Lying':
            lying += 1
        elif data['estimate'][k] == 'Standing':
            standing += 1
        elif data['estimate'][k] == 'Wheeling':
            wheeling += 1
        elif data['estimate'][k] == 'Sitting':
            sitting += 1
        elif data['estimate'][k] == 'Misc' or data['estimate'][k] == 'Trash':
            misc += 1
            
        
        if (data['time'][k] - inicial_time >= clips_range):
            if (walking/float(len(X[clips]))) > 0.8 or lying/float(len(X[clips])) > 0.8 or wheeling/float(len(X[clips])) > 0.8 or standing/float(len(X[clips])) > 0.8 or sitting/float(len(X[clips])) > 0.8:

                state.append( data['estimate'][k-10] )
                
                #Add more features
                features.append([])
                features[clipsF].append( sum(X[clips])/len(X[clips]))
                features[clipsF].append( sum(Y[clips])/len(Y[clips]))
                features[clipsF].append( sum(Z[clips])/len(Z[clips]))

                TX,TY,TZ = np.array(X), np.array(Y), np.array(Z)
                features[clipsF].append( TX[clips].std())
                features[clipsF].append( TY[clips].std())
                features[clipsF].append( TZ[clips].std())

                
                #Label the target with the respective predominant action in the clip
                if walking ==  max(misc,lying, standing, wheeling, sitting,walking):
                    target.append(1)
                elif lying ==  max(misc,lying, standing, wheeling, sitting,walking):
                    target.append(0)
                elif standing ==  max(misc,lying, standing, wheeling, sitting,walking):
                    target.append(2)
                elif wheeling ==  max(misc,lying, standing, wheeling, sitting,walking):
                    target.append(3)
                elif sitting ==  max(misc,lying, standing, wheeling, sitting,walking):
                    target.append(4)

                clipsF = clipsF + 1

                
            walking, notwalking, misc,lying, standing, wheeling, sitting = 0 , 0, 0,0,0,0,0
            clips = clips + 1
            X.append([])
            Y.append([])
            Z.append([])
            
            inicial_time = data['time'][k]
            
        #print data['itime'][k] - inicial_time
        k -= start_point
        X[clips].append( data['x'][k + start_point] )
        Y[clips].append( data['y'][k + start_point] )
        Z[clips].append( data['z'][k + start_point] )
        k += start_point

    features = np.array(features)

    target = np.array(target)

    return (features,target)



########################################################################################





    

#Get the clips of features and target from a CSV file
features, target = get_features_target('/Users/Ranulfo/HMM/Pre/Subject_1/Subject_1_Pre_sized1_modifiedlabeled.csv')


#Start the logistic Regression
logreg = linear_model.LogisticRegression(C= 1000000, penalty='l1', dual=False, tol = 0.00001)

#Train the data (features) with the target
logreg.fit(features, target)

#Get the probability of each state from the prediction of LR
Resul = logreg.predict_proba(features)

#Get the predicted states of the LR
ResulM = logreg.predict(features)

#Get the probability of success LR
score = comp(ResulM, target)
#score2 = comp(Resul, target)
print "LR ="
print score
#print score2





########################################################################################

#Probability Matrix =====> It's wrong !! --- Fix the values
#Every row must sum = 1
#Each row means one state stating with state 0

a = 0.897
b = 0.100
c = 0.0014999999999999458
e = 0.001
d = 0.050

##EX transitions_prob = np.mat([row0 = [a,c,d,c,d], row1 = [ e,a,b,e,e], row2 = [c,d,a,c,d] , row3 = [d,c,c,a,d] , row4  [d,c,d,c ,a]])

transitions_prob = np.mat([[a,c,d,c,d],[ e,a,b,e,e], [c,d,a,c,d] , [d,c,c,a,d] , [d,c,d,c ,a]])


HMM = GaussianHMM(n_components = 5,  covariance_type= "diag", transmat= transitions_prob)


#
#Must always fit the obs data before change means and covars
#
HMM.fit([Resul])

HMM.means_ = np.identity(5)

HMM.covars_ = 0.2*np.ones((5,5))

#Use of LR probability to predict the states.
HResul = HMM.predict(Resul)

#Get the probability of success HMM
Hscore = comp(HResul,target)

#print HResul

print "HMM = "
print Hscore




##      Here is just for writing the results inside a CSV file

########################################################################################

#target = np.where(target == 0, 'Lying')

target = [ 'Lying' if x == 0 else x for x in target]
target = ['Walking' if x == 1 else x for x in target]
target = ['Standing' if x == 2 else x for x in target]
target = ['Wheeling' if x == 3 else x for x in target]
target = ['Sitting' if x == 4 else x for x in target]

HResul = ['Lying' if x == 0 else x for x in HResul]
HResul = ['Walking' if x == 1 else x for x in HResul]
HResul = ['Standing' if x == 2 else x for x in HResul]
HResul = ['Wheeling' if x == 3 else x for x in HResul]
HResul = ['Sitting' if x == 4 else x for x in HResul]

ResulM = ['Lying' if x == 0 else x for x in ResulM]
ResulM = ['Walking' if x == 1 else x for x in ResulM]
ResulM = ['Standing' if x == 2 else x for x in ResulM]
ResulM = ['Wheeling' if x == 3 else x for x in ResulM]
ResulM = ['Sitting' if x == 4 else x for x in ResulM]

        #0 -> Lying
        #1 -> Walking
        #2 -> Standing
        #3 -> Wheeling
        #4 -> Sitting


#Write the results of HMM LR and Target inside a CSV file
df = { 'HMM' : HResul, 'LR' : ResulM,'Target' : target}

nData = pd.DataFrame(df)

nData.to_csv('/Users/Ranulfo/HMM/TestComplete.csv')


#######################################################################################





