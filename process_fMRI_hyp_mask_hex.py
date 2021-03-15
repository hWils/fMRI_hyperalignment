# processing fMRI data
# find path to fmri files
# find path to SPM.MAT files


import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from nibabel import save
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.experimental_paradigm import  BlockParadigm # EventRelatedParadigm
from os import mkdir, path
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import sklearn
import glob
import nibabel
import os
from hw_edits.stats_hw import align
#from nltools.stats import align
import matplotlib.pyplot as plt
import warnings
import json
from mvpa2.suite import *
import nibabel as nib
from nilearn.masking import apply_mask as nimask


# submethod of extracting the perception onsets and durations
def block_to_stimuli_onset(block_ons, dur, stim_num = 36):
    perception_ons = np.zeros(stim_num)
    perception_duration = 7* (np.ones(stim_num))
    i = 0
    for block in block_ons:
        for stim in range(4):
            if stim == 0:
                perception_ons[i] = block
            else:
                perception_ons[i] = block + (dur*stim)
            i+=1
    return perception_ons, perception_duration

# create the correct filepath for retreving the 4D fmri file for a particular subject and condition 
def condition_type_filepath(condition, subject, types):
    main_filepath = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//Imagine_' + subject
    if condition == 'Faces':
        filepath =   main_filepath + '//Faces//'
        if types =='spm':
            filepath = filepath + 'SPM.mat'
        elif types == '4D':
           # print("here                         : ",filepath)
            filepath = filepath + 'Segmented//Motion_corrected//4D.nii.gz'
    elif condition == 'Places':
        filepath =   main_filepath + '//Places//'
        if types =='spm':
            filepath = filepath + 'SPM.mat'
        elif types == '4D':
            filepath = filepath + 'Segmented//Motion_corrected//4D.nii.gz'
            print("4d filepath is ", filepath)
    return filepath

# given an spm file path, extract the onset and duration times for each part of the experiment, including control conditions
def getOnsetDurationDurImgPercSPM(condition_filepath, stim_num = 36):
    #get imagination onset and duration
    mat = spio.loadmat(condition_filepath, squeeze_me=True)
    spm = mat['SPM'][()]

    imagination = spm['Sess']['U'].item()['ons'][0]
    imagination_dur = spm['Sess']['U'].item()['dur'][0]

    name = spm['Sess']['U'].item()['ons'][1]
    name_dur = spm['Sess']['U'].item()['dur'][1]

    look_text = spm['Sess']['U'].item()['ons'][2]
    look_text_dur = spm['Sess']['U'].item()['dur'][2]

    imagine_text = spm['Sess']['U'].item()['ons'][4]
    imagine_text_dur = spm['Sess']['U'].item()['dur'][4]

    control_imagine = spm['Sess']['U'].item()['ons'][5]
    control_imagine_dur = spm['Sess']['U'].item()['dur'][5]

    control_name = spm['Sess']['U'].item()['ons'][6]
    control_name_dur = spm['Sess']['U'].item()['dur'][6]

    look_control_text = spm['Sess']['U'].item()['ons'][7]
    look_control_text_dur = spm['Sess']['U'].item()['dur'][7]

    control_imagine_text = spm['Sess']['U'].item()['ons'][9]
    control_imagine_text_dur = spm['Sess']['U'].item()['dur'][9]

    block_perception_ons = spm['Sess']['U'].item()['ons'][3]
    control_block_perception_ons = spm['Sess']['U'].item()['ons'][8]
    percieve, percieve_dur = block_to_stimuli_onset(block_perception_ons,dur=7)
    control_perceive, control_perceive_dur = block_to_stimuli_onset(control_block_perception_ons,dur=7) 

    conditions = [['imagination']*stim_num,['name']*stim_num,['look_text']*9,['percieve']*stim_num,
                  ['imagine_text']*9,['control_imagine']*stim_num,['control_name']*stim_num,
                  ['look_control_text']*9,['control_perceive']*stim_num,['control_imagine_text']*9]

    onsets = [imagination,name,look_text,percieve,
            imagine_text,control_imagine,control_name,look_control_text, control_perceive,control_imagine_text]

    durations = [imagination_dur,name_dur,look_text_dur,percieve_dur,
            imagine_text_dur,control_imagine_dur,control_name_dur,look_control_text_dur, control_perceive_dur,control_imagine_text_dur]
    return conditions, onsets,durations 




# for exemplar level, find onsets and durations given a particular filepath and conditions
def getOnsetExemplarLevelM(condition_filepath, task, stim_num = 36):
    #get imagination onset and duration
    mat = spio.loadmat(condition_filepath, squeeze_me=True)
    spm = mat['SPM'][()]
    exemplars = np.arange(1,37)
    exemplar_condition_names = []
    exemplar_onsets = []
    exemplar_durations =[]

    if task == 'imagine':
        onset = spm['Sess']['U'].item()['ons'][0]
        duration = spm['Sess']['U'].item()['dur'][0]
    elif task == 'perceive':
        block_perception_ons = spm['Sess']['U'].item()['ons'][3]
        onset, duration = block_to_stimuli_onset(block_perception_ons,dur=7)

    for i in range(36):
        name = task + '_' + str(exemplars[i])
        exemplar_onsets.append(onset[i])
        exemplar_durations.append(duration[i])
        exemplar_condition_names.append(name)
        print("onsets, durations: ", len(exemplar_onsets), " ", len(exemplar_durations))

    return exemplar_condition_names, exemplar_onsets,exemplar_durations 



# Input = condition, fMRI .nii files, spm .mat files. Output = designmatrix.png, GLM model, contrast images .nii
# Takes in the .nii files for each dynamic at the subject level, extracts the timings/condition from the .mat file
#  creates and saves a png of the design matrix. Then obtains the 4D fMRI data, and fits a GLM
# defines contrasts, then carries out contrasts, which produces a contrast .nii file and saves

def first_level_analysis(subject,faceOrplace, stim_num = '36'):

    if faceOrplace != 'Faces' and faceOrplace != 'Places':
        print("You need to enter either Faces or Places as a parameter to this function")
    else:
        print("Conducting single_level_analysis for " + faceOrplace + "condition, for subject number "+subject)
    # get filepaths for faces and places
    filepath = condition_type_filepath(faceOrplace, subject, types = 'spm')
    print(filepath)

    # onsets and durations for both places and faces for a given subject
    condition_list, onset, duration = getOnsetDurationDurImgPercSPM(filepath) ##########

    # flatten lists
    flattened_conditions =list(chain.from_iterable(condition_list))
    flattened_onsets =  list(chain.from_iterable(onset))
    flattened_durations =  list(chain.from_iterable(duration))
    print("These two numbers should be equal, if not there is an error ", len(flattened_conditions), len(flattened_onsets))
    paradigm = BlockParadigm(con_id= flattened_conditions, onset=flattened_onsets, duration=flattened_durations)

    n_scans = 330 #dynamics
    tr = 3
    
    
    frametimes = np.arange(0, n_scans*tr,tr)
    hrf_model = 'canonical'
    drift_model = "cosine"
    hfcut = 128 # what does this refer to? bandpass
    design_matrix = make_dmtx(frametimes,paradigm,hrf_model=hrf_model, drift_model = drift_model, hfcut=hfcut)
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    ax.set_title('Design matrix')
    resultDir = 'C://Users//hlw69//Documents//fMRI_Exeter//masked_results//'
    filename = subject + "_" + faceOrplace 
    write_dir = path.join(resultDir, filename) 
    if not path.exists(write_dir):
        mkdir(write_dir)
    plt.savefig(path.join(write_dir, 'design_matrix.png'))

    # obtain the neural data, this will be in a 4D format nii.gz
    spm4Dfilepath = condition_type_filepath(faceOrplace, subject, types = '4D')
    
    img_4d = nib.load(spm4Dfilepath)
    maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//masks_roi//r_testfirstAttemptDilated.nii'
    roi_mask = nib.load(maskDirectory)
   # masked_data = nimask(img_4d,roi_mask) # this returns a 2D array
    

    ##  Create the fmri linear model
    fmri_glm = FMRILinearModel(img_4d, design_matrix.matrix,
                            mask= roi_mask)
    fmri_glm.fit(do_scaling=True, model='ar1')
    

    contrasts = {}
    contrasts = {
    "imagination": np.zeros(25),
    "perception": np.zeros(25),
    "Perc_Imag": np.zeros(25), 'Imag_Perc':np.zeros(25)}
    contrasts['imagination'][4] = 1
    contrasts['perception'][9] = 1
    contrasts['Perc_Imag'][9] = 1
    contrasts['Perc_Imag'][4] = -1
    contrasts['Imag_Perc'][4] = 1
    contrasts['Imag_Perc'][9] = -1

    print("length of contrasts = ", len(contrasts))
    print("Contrasts include : ", contrasts)
    print('Computing contrasts...')
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
       # print('  Contrast % 2i out of %i: %s' %
#            (index + 1, len(contrasts), contrast_id))
        # save the z_image
        image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
        z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
        save(z_map, image_path)

        # Create snapshots of the contrasts
        vmax = max(-z_map.get_data().min(), z_map.get_data().max())
        plot_map(z_map.get_data(), z_map.get_affine(),
                cmap=cm.cold_hot, vmin=-vmax, vmax=vmax,
                slicer='z', black_bg=True, threshold=2.5,
                title=contrast_id)
        plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

    print("All the  results were witten in %s" % write_dir)



 #    

def first_level_exemplar(subject,faceOrplace, imagineOrPerceive):
        
    if faceOrplace != 'Faces' and faceOrplace != 'Places':
        print("You need to enter either Faces or Places as a parameter to this function")
    else:   
        print("Conducting single_level_analysis at the exemplar level for " + faceOrplace + "condition, for subject number "+subject)
    # get filepaths for faces and places
    filepath = condition_type_filepath(faceOrplace, subject, types = 'spm')
    
    condition_list, onset, duration= getOnsetExemplarLevelM(filepath,task=imagineOrPerceive) ##
    print(condition_list)

    paradigm = BlockParadigm(con_id=condition_list, onset=onset, duration=duration)

    n_scans = 330 #dynamics
    tr = 3
    frametimes = np.arange(0, n_scans*tr,tr)
    hrf_model = 'canonical'
    drift_model = "cosine"
    hfcut = 128 #  bandpass
    design_matrix = make_dmtx(frametimes,paradigm,hrf_model=hrf_model, drift_model = drift_model, hfcut=hfcut)
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    ax.set_title('Design matrix')
    resultDir = 'C://Users//hlw69//Documents//fMRI_Exeter//masked_results//'
    filename = subject + "_" + faceOrplace + '//'
    write_dir = path.join(resultDir, filename) 
    if not path.exists(write_dir):
        mkdir(write_dir)
    write_dir = path.join(write_dir,(faceOrplace + '_contrasts//'))
    print(write_dir)
    if not path.exists(write_dir):
        mkdir(write_dir)
    write_dir = path.join(write_dir,(imagineOrPerceive))
    print(write_dir)
    if not path.exists(write_dir):
        mkdir(write_dir)
    plt.savefig(path.join(write_dir, 'design_matrix_exemplar_level.png'))

    spm4Dfilepath = condition_type_filepath(faceOrplace, subject, types = '4D')
    print(spm4Dfilepath) #

    
    ####################################################################################   ADD MASK HERE?? ################


    
    
    
    
     ##  Create the fmri linear model
   # fmri_glm = FMRILinearModel(spm4Dfilepath, design_matrix.matrix,
    fmri_glm = FMRILinearModel(img_4d, design_matrix.matrix,
                            mask= 'compute')
    fmri_glm.fit(do_scaling=True, model='ar1')
    

    identity = np.identity(51, dtype=int).tolist()
    contrasts = {}
    for x in range(1,stim_num+1):
        contrasts[x]=np.array(identity[x-1])#
    print(len(contrasts))
    print(contrasts.keys())
    #print(contrasts.get('1'))
   # print(contrasts)

    print('Computing contrasts...')
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
            (index + 1, len(contrasts), contrast_id))
        # save the z_image
        image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
        z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
        save(z_map, image_path)

        # Create snapshots of the contrasts
        vmax = max(-z_map.get_data().min(), z_map.get_data().max())
        plot_map(z_map.get_data(), z_map.get_affine(),
                cmap=cm.cold_hot, vmin=-vmax, vmax=vmax,
                slicer='z', black_bg=True, threshold=2.5,
                title=contrast_id)
        plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

    print("All the  results were witten in %s" % write_dir)
    #plt.show()
    return contrasts



# get all the contrast files for faces
def getContrastFilesMask(faceOrPlace, subject, condition, mask):
    print(faceOrPlace, "condition")
    directory = 'C://Users//hlw69//Documents//fMRI_Exeter//results//'+ str(subject)+ '_'+faceOrPlace+'//'+faceOrPlace+'_contrasts//'+ condition + '//'
    os.chdir(directory) 
    contrastFiles = []
    for file in glob.glob("*.nii"):
        contrastFiles.append(file)
    os.listdir()
    contrastFiles.sort(key= lambda x: float(x.strip('_z_map.nii')))
    maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//masks_roi//rdilation_one_3D.nii'
    
    # maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//Imagine_'+str(subject)+'//'+mask+'//mask.img'
    mask = nibabel.load(maskDirectory).get_data()
    mask = mask.astype(bool)
    return contrastFiles, directory, mask
        

    
def applyMaskData(mask, contrastFiles, directory):
    contrast_array = []
    for contrast in contrastFiles:
        img_nifti = nibabel.load(directory+contrast)
        img = img_nifti.get_data()
        X = img[mask].T
        contrast_array.append(list(X))
        print("X : ", type(X))
        print("contrast_array : ", type(contrast_array))
    return contrast_array




import copy
def logisticRegressionExemplar(faceOrPlace, perception, imagine):
    print("Are they equal mew", np.array_equal(perception, imagine))
    X_train = copy.deepcopy(perception)
    X_test = copy.deepcopy(imagine)

    lr = LogisticRegression(random_state=0)
   # X_train =  train_data# perception
    #X_test =    test_data # imagine
    y_train = np.arange(36)
    y_test = np.arange(36)

    lr.fit(X_train, y_train)
    print("score on training perception data ", lr.score(X_train, y_train))
    score = lr.score(X_test, y_test)
    print("score on imagine test data ", score)
    predictions = lr.predict(X_test)
    print(f"Predictions for imagine are : {predictions}")
    return score



    

def logisticRegression_a_Or_b(a_data, b_data):
    print(a_data.shape, b_data.shape)
    a_vs_b = np.concatenate((a_data, b_data), axis=0)
    target_P_F = np.zeros(72)
    target_P_F[36:72] +=1
  #  print("the target and data length should be equal ",len(target_P_F), len(a_vs_b))
    X_train, X_test, y_train, y_test = train_test_split(a_vs_b, target_P_F, test_size=0.1, random_state=8, shuffle=True)
   # lr = LogisticRegressionCV(cv = 16, penalty = 'l2',random_state=6)
    lr = LogisticRegression(penalty = 'l2',random_state=6)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    # training accuracy  # print out.
    # print learning curve
    # is this the same for each person??
    # logistic regression is last layer in NN.. softmax equivalent
  #  print("probabilities on test data ",lr_P_F.predict_proba(X_test))
   # print("accuracy on training data: ", lr_P_F.score(X_train, y_train))
   # print("parameters :", lr_P_F.get_params())
   # print("coefficients ", lr_P_F.coef_)
    print(f"accuracy on test data : {score}")
    print("what type is score ",type(score))
   # print("CATEGORY PREDICTIONS :", predictions)
    return score
    


"""
stim_num = 36
subject = 34
condition = 'perceive'




################################################   REDOING WITH ROI MASK
for subject in range(30,59):
    if subject == 39:
        print("skip 39")
    else:
        first_level_analysis(str(subject), 'Faces')
        first_level_analysis(str(subject), 'Places')
        
first_level_exemplar       
######################################################################  here tomorrow

     """   
# after this... first level exemplar??/




"""
for subject in range(30,59):
    if subject == 39:
        print("skip 39")
    else:
        print(subject)
        subject = str(subject)
        contrast = first_level_exemplar(subject, 'Faces', 'perceive')
        contrast = first_level_exemplar(subject, 'Places', 'perceive')
        contrast = first_level_exemplar(subject, 'Faces', 'imagine')
        contrast = first_level_exemplar(subject, 'Places', 'imagine')


# all files are now created for the first level analysis, should not need to get exemplar level files
# when performing logistic regression
# all need is to extract the contrast files.. such as perception
# then convert  them into the correct format.. masked and array form.
a_ContrastFiles, a_Directory, a_Mask = getContrastFilesMask('Faces', subject, 'perceive', mask = 'Faces')
b_ContrastFiles, b_Directory, b_Mask = getContrastFilesMask('Places', subject, 'perceive', mask = 'Faces')
#apply mask - and needs to be in array form for logistic regression
a_contrast_arrays = np.array(applyMaskData(a_Mask, a_ContrastFiles, a_Directory))
b_contrast_arrays = np.array(applyMaskData(b_Mask, b_ContrastFiles, b_Directory))

"""


# each subject one by one, creates perception contrast arrays for faces and places - should do this for imagination too, then store them somewhere

dict_faces_perception = {}
dict_places_perception = {}


# need to run this to load up contrasts, they are too big to be saved into csv or json files
"""
for subject in range(30,59):
    if subject == 39:
        dict_faces_perception['39'] = 'NULL'
        dict_places_perception['39'] = 'NULL'
        print("skip")
    else:
        print("subject is ", subject)
        a_ContrastFiles, a_Directory, a_Mask = getContrastFilesMask('Faces', subject, 'perceive', mask = 'Faces')
        b_ContrastFiles, b_Directory, b_Mask = getContrastFilesMask('Places', subject, 'perceive', mask = 'Faces')

      #  apply mask - and needs to be in array form for logistic regression
        Perceive_Faces_contrast_arrays = applyMaskData(a_Mask, a_ContrastFiles, a_Directory)
        Perceive_Places_contrast_arrays = applyMaskData(b_Mask, b_ContrastFiles, b_Directory)
        dict_faces_perception[str(subject)] = Perceive_Faces_contrast_arrays
        dict_places_perception[str(subject)] = Perceive_Places_contrast_arrays





import json

with open('dict_faces_perception_roi_hex.json', 'w') as fp:
    json.dump(dict_faces_perception, fp)
with open('dict_faces_perception_roi_hex.json', 'r') as fp:
    contrasts_dict_faces_perception = fp.readline()
    contrasts_dict = json.loads(contrasts_dict_faces_perception)
fp.close()
    
"""
"""

dict_faces_imagination = {}
dict_places_imagination = {}

imagine_scores_subject = []
for subject in range(30,59):
    if subject == 39:
        dict_faces_imagination['39'] = 'NULL'
        dict_places_imagination['39'] = 'NULL'
        print("skip")
    else:
        print("subject is ", subject)
        a_ContrastFiles, a_Directory, a_Mask = getContrastFilesMask('Faces', subject, 'imagine', mask = 'Faces')
        b_ContrastFiles, b_Directory, b_Mask = getContrastFilesMask('Places', subject, 'imagine', mask = 'Faces')
        #apply mask - and needs to be in array form for logistic regression, but not yet
        Imagine_Faces_a_contrast_arrays = applyMaskData(a_Mask, a_ContrastFiles, a_Directory)
        Imagine_Places_b_contrast_arrays = applyMaskData(b_Mask, b_ContrastFiles, b_Directory)
        dict_faces_imagination[str(subject)] = Imagine_Faces_a_contrast_arrays #faces
        dict_places_imagination[str(subject)] = Imagine_Places_b_contrast_arrays #places






dict_perc_f_vs_p = {}
dict_imag_f_vs_p = {}


dict_exemplar_face_i={}
dict_exemplar_place_i ={}
dict_exemplar_face_i_from_p={}
dict_exemplar_place_i_from_p ={}

dict_exemplar_p= {}
dict_exemplar_i = {}

for subject in range(30,59):
    if subject == 39:
        print("skip 39")
    else:
        print(subject)
        score = logisticRegressionExemplar(faceOrPlace="Faces", perception = dict_faces_perception[str(subject)], imagine = dict_faces_imagination[str(subject)])
        dict_exemplar_face_i_from_p[str(subject)] = score
        score = logisticRegressionExemplar(faceOrPlace="Places", perception = dict_places_perception[str(subject)], imagine = dict_places_imagination[str(subject)])
        dict_exemplar_place_i_from_p[str(subject)] = score
        
     # faces vs places 
     #   scorePerception = logisticRegression_a_Or_b(np.array(dict_faces_perception[str(subject)]), np.array(dict_places_perception[str(subject)]))
     #   dict_perc_f_vs_p[str(subject)] = scorePerception 
     #   scoreImagination = logisticRegression_a_Or_b(np.array(dict_faces_imagination[str(subject)]), np.array(dict_places_imagination[str(subject)]))
     #   dict_imag_f_vs_p[str(subject)] = scoreImagination



print(dict_exemplar_place_i_from_p.values())


print("perception faces vs places : ", dict_perc_f_vs_p.values())
print("imamgination faces vs places : ", dict_imag_f_vs_p.values())

from matplotlib import pyplot as plt
bins = np.arange(0, 1, 0.1) # fixed bin size
plt.hist(dict_imag_f_vs_p.values(), bins=bins, alpha=0.5, edgecolor='black')
plt.title('Imagination face vs place categorisation')
plt.xlabel('Classification accuracy')
plt.ylabel('count')

plt.show()










subject30 = {dict_faces_perception['30'],dict_places_perception['30'],}


import csv
csvfile = "dict_places_perception.csv"

with open(csvfile, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=dict_places_perception.keys())
    writer.writeheader()
    for data in dict_places_perception:
        print("row written)")
        writer.writerow(data)
print("done writing")
csvfile.close()


data = csv.DictReader(open(csvfile))
print("CSV file as a dictionary:\n")
for row in data:
   print(row)









print("dict ffaces perception ", type(dict_contrasts['faces_perception']['30'][0]))


    
print(dict_faces_perception.keys()) 


print("hi")
print("where ", contrasts_dict['30'])
      
      
      

      
print(dict_faces_perception['30'][0])    
      
      
## contrasts for perception faces + places, with each exemplar
# leave out entire subject as the test set,
# across rather than within-subject classification
# chose the amount of features - 50 chen et al 2015
# 



       


#91, 109, 91 = 2mm voxels  MNI default
# 79, 95, 69, what the exeter dataset is...

# has become 2D at this point, needs to be in 3D format to apply mask
# run the analyses again but on 4D.nii and apply mask part way through process

import nibabel as nib
img = nib.load('C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//Imagine_31//Faces//Segmented//Motion_corrected//4D.nii')
maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//masks_roi//firstAttemptDilated.nii'
#roi_mask = nibabel.load(maskDirectory).get_data()
maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//masks_roi//r_testfirstAttemptDilated.nii'

roi_mask = nib.load(maskDirectory)
print(roi_mask.shape, "mask")
print(img.shape, "img")
"""        
# observations by features
# 2,  199547



#with open('C:/Users/hlw69/Documents/fMRI_Exeter/results/58_Places/Places_contrasts/imagine/dict_faces_perception_roi_hex.json') as fp:
with open('dicts/dict_faces_perception_roi_hex.json') as fp:
    dict_faces_perception = json.load(fp)

warnings.simplefilter('ignore')
all_data = []
for subject in range(30,59):
    if subject == 39:
        print("skip 39")
    else:
        print(subject)
        data = np.array(dict_faces_perception[str(subject)])
        data = data.astype(np.float32)
        scaled = sklearn.preprocessing.minmax_scale(data, feature_range=(0, 1), axis=0, copy=True)
        print(np.min(scaled), np.max(scaled))
        all_data.append(scaled)

hyperalign = align(all_data[0:2][0:5], method = 'procrustes')






"""
from numba import jit, cuda
import numba
from numba import vectorize, guvectorize

#vectorize attempt
def do_hyper(array_all_data):
    cat = array_all_data + array_all_data
    return cat

hyper_vectorized = vectorize("float32[:,:,:](float32[:,:,:])", 
                          target="cuda")(do_hyper)


# guvectorize attempt as need array input - this finally works, open from anaconda navigator thoughs
@guvectorize(['void(float64[:,:], float64[:,:], float64[:,:])'],
             '(m,n),(n,p)->(m,p)',target='cuda')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
                
                
@guvectorize(['void(float32[:,:,:], float32[:,:,:], float32[:,:,:])'],
             '(),()->()',target='cuda')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

"""

"""

print(cuda.gpus)
threadsperblock = 32
blockspergrid = (len(all_data) + (threadsperblock - 1)) // threadsperblock

print(np.asarray(all_data).shape)
array_all_data = np.asarray(all_data)
#GUvectorise accepts array arguments


#  a kernel cannot have any return value
@cuda.jit("float32[:,:,:](float32[:,:,:])", device=True)
def do_hyper(array_all_data):
    cat = array_all_data + array_all_data
   # hyperalign = align(all_data[0:2][0:5], method = 'procrustes')

threadsperblock = 1024
blockspergrid = (28 + (threadsperblock - 1)) // threadsperblock
do_hyper[blockspergrid, threadsperblock](all_data) 

"""
#@vectorize([],target="cuda")
#@numba.vectorize(["float32[:,:](float32[:,:,:])"],nopython=True, target ="cuda")
#def do_hyper(array_all_data):
 

#hyperalign = array_all_data(all_data[0:2][0:5], method = 'procrustes')

# is using ram or gpu memory
# made it to d when changed to gesvd!



"""
    
# FACES VS PLACES WITHIN SUBJECT 
face_contrasts = first_level_exemplar(str(subject),faceOrplace = 'Faces', imagineOrPerceive = condition)
place_contrasts = first_level_exemplar(str(subject),faceOrplace = 'Places', imagineOrPerceive = condition)

FaceContrastFiles, faceDirectory, FaceMask = getContrastFilesMask('Faces', subject, condition)
PlaceContrastFiles, placeDirectory, PlaceMask = getContrastFilesMask('Places', subject, condition)

face_contrast_arrays = applyMaskData(FaceMask, FaceContrastFiles, faceDirectory)
place_contrast_arrays = applyMaskData(FaceMask, PlaceContrastFiles, placeDirectory) # using face mask to ensure same shape
    

#needs to be in array form for logistic regression
face_image_arrays = np.array(face_contrast_arrays)
place_image_arrays = np.array(place_contrast_arrays)


#logisticRegressionExemplar("Faces", face_image_arrays)
#logisticRegressionExemplar("Places", place_image_arrays)

"""
"""
logisticRegressionFaceOrPlace(face_image_arrays, place_image_arrays)

# compare two suubjects - see if contrast files are identical somehow, seems fine
# print lr __f


"""
"""
directory33 = 'C://Users//hlw69//Documents//fMRI_Exeter//results//33_Faces//Faces_contrasts//perceive//'
os.chdir(directory33) 
contrastFiles33 = []
for file in glob.glob("*.nii"):
    contrastFiles33.append(file)
os.listdir()
contrastFiles33.sort(key= lambda x: float(x.strip('_z_map.nii')))

directory34 = directory = 'C://Users//hlw69//Documents//fMRI_Exeter//results//34_Faces//Faces_contrasts//perceive//'
os.chdir(directory34) 
contrastFiles34 = []
for file in glob.glob("*.nii"):
    contrastFiles34.append(file)
os.listdir()
contrastFiles34.sort(key= lambda x: float(x.strip('_z_map.nii')))

img30 = nibabel.load(directory33 + contrastFiles33[0])
img38 = nibabel.load(directory34+contrastFiles34[0])
print("lower level " , np.array_equal(img30.get_fdata(),img38.get_fdata()))
print(np.max(img30.get_fdata()), " split ", np.max(img38.get_fdata()))
"""




"""

first_level_analysis('38','Places')
first_level_analysis('30','Places')



filepath = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//Imagine_33//Places//Segmented//Motion_corrected//'
filepath1 = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//Imagine_34//Places//Segmented//Motion_corrected//'


directory33C = filepath
os.chdir(directory33C) 
Files33C = []
for file in glob.glob("*.nii.gz"):
    if '4D' in file:
        Files33C.append(file)
        print(file)
os.listdir()
directory34C = filepath1
os.chdir(directory34C) 
Files34C = []
for file in glob.glob("*.nii.gz"):
    if '4D' in file:
        Files34C.append(file)
        print(file)
os.listdir()
img30 = nibabel.load(filepath + Files33C[0])
img38 = nibabel.load(filepath1+Files34C[0])
print(img30.get_fdata()[5][6], " split ", img38.get_fdata()[5][6])
#the 4D files are the same..what about the non4D files
print("lower level " , np.array_equal(img30.get_fdata(),img38.get_fdata()))

# at least the 4D is different, should test this for cases other than 38 and 30

print(np.array_equal(contrastFiles33,contrastFiles34))
"""