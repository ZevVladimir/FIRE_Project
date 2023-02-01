import numpy as np

file = '/home/zeevvladimir/Personal_Project/TNG300_RF_data-20221026T024254Z-001/TNG300_RF_data/'
Group_M_Mean200_dm = np.load(file+'Group_M_Mean200_dm.npy')*1e10
print(Group_M_Mean200_dm.shape)
Group_M_Mean200_dm = Group_M_Mean200_dm[:286378]

GroupPos_dm = np.load(file+'GroupPos_dm.npy')/1000
GroupPos_dm = GroupPos_dm[:286378]

GroupConc_dm = np.load(file+'GroupConc_dm.npy')
GroupConc_dm = GroupConc_dm[:286378]

GroupEnv_dm = np.load(file+'GroupEnv_dm.npy')
GroupEnv_dm = GroupEnv_dm[:286378]

GroupEnvAnn_dm = np.load(file+'GroupAnnEnv_R5_dm.npy')  
GroupEnvAnn_dm = GroupEnvAnn_dm[:286378]

GroupEnvTH_dm = np.load(file+'GroupEnv_dm_TopHat_1e11mass.npy')[:,15] #already masked for Mhalo>1e11
GroupEnvTH_dm = GroupEnvTH_dm[:277481]

GroupSpin_dm = np.load(file+'GroupSpin_dm.npy')
GroupSpin_dm = GroupSpin_dm[:286378]

GroupNsubs_dm = np.load(file+'GroupNsubs_dm.npy')
GroupNsubs_dm = GroupNsubs_dm[:286378]

GroupVmaxRad_dm = np.load(file+'GroupVmaxRad_dm.npy')
GroupVmaxRad_dm = GroupVmaxRad_dm[:286378]

Group_SubID_dm = np.load(file+'GroupFirstSub_dm.npy') #suhalo ID's
Group_SubID_dm = Group_SubID_dm[:286378]

Group_Shear_dm = np.load(file+'GroupShear_qR_dm_1e11Mass.npy') #already,masked for Mhalo>1e11
Group_Shear_dm = Group_Shear_dm[:277481]

SubVdisp_dm = np.load(file+'SubhaloVelDisp_dm.npy')
SubVdisp_dm = SubVdisp_dm[:286378]

SubVmax_dm = np.load(file+'SubhaloVmax_dm.npy')
SubVmax_dm = SubVmax_dm[:286378]

SubGrNr_dm = np.load(file+'SubhaloGrNr_dm.npy') #Index into the Group table of the FOF host/parent of Subhalo
SubGrNr_dm = SubGrNr_dm[:286378]

SubhaloPos_dm = np.load(file+'SubhaloPos_dm.npy')/1000
SubhaloPos_dm = SubhaloPos_dm[:286378]

count_dm = np.load(file+'GroupCountMass_dm.npy')
count_dm = count_dm[:286378]

cent_count_dm = np.load(file+'GroupCountCentMass_dm.npy')
cent_count_dm = cent_count_dm[:286378]

sat_count_dm = count_dm-cent_count_dm
sat_count_dm = sat_count_dm[:286378]

GroupEnvTH_1_3 = np.load(file+'GroupEnv_dm_TopHat_1e11mass.npy')[:,7] #env at 1.3 Mpc
GroupEnvTH_1_3 = GroupEnvTH_1_3[:277481]
GroupEnvTH_2_5 = np.load(file+'GroupEnv_dm_TopHat_1e11mass.npy')[:,11] #env at 2.6 Mpc
GroupEnvTH_2_5 = GroupEnvTH_2_5[:277481]

Group_R_Mean200_dm = np.load(file+'Group_R_Mean200_dm.npy')
Group_R_Mean200_dm = Group_R_Mean200_dm[:286378]


np.save(file = (file + "new_Group_M_Mean200_dm"), arr = Group_M_Mean200_dm)
np.save(file = (file + "new_GroupPos_dm"), arr = GroupPos_dm)
np.save(file = (file + "new_GroupConc_dm"), arr = GroupConc_dm)
np.save(file = (file + "new_GroupEnv_dm"), arr = GroupEnv_dm)
np.save(file = (file + "new_GroupEnvAnn_dm"), arr = GroupEnvAnn_dm)
np.save(file = (file + "new_GroupEnvTH_dm"), arr = GroupEnvTH_dm)
np.save(file = (file + "new_GroupSpin_dm"), arr = GroupSpin_dm)
np.save(file = (file + "new_GroupNsubs_dm"), arr = GroupNsubs_dm)
np.save(file = (file + "new_GroupVmaxRad_dm"), arr = GroupVmaxRad_dm)
np.save(file = (file + "new_Group_SubID_dm"), arr = Group_SubID_dm)
np.save(file = (file + "new_Group_Shear_dm"), arr = Group_Shear_dm)
np.save(file = (file + "new_SubVdisp_dm"), arr = SubVdisp_dm)
np.save(file = (file + "new_SubVmax_dm"), arr = SubVmax_dm)
np.save(file = (file + "new_SubGrNr_dm"), arr = SubGrNr_dm)
np.save(file = (file + "new_SubhaloPos_dm"), arr = SubhaloPos_dm)
np.save(file = (file + "new_count_dm"), arr = count_dm)
np.save(file = (file + "new_cent_count_dm"), arr = cent_count_dm)
np.save(file = (file + "new_sat_count_dm"), arr = sat_count_dm)
np.save(file = (file + "new_GroupEnvTH_1_3"), arr = GroupEnvTH_1_3)
np.save(file = (file + "new_GroupEnvTH_2_5"), arr = GroupEnvTH_2_5)
np.save(file = (file + "new_Group_R_Mean200_dm"), arr = Group_R_Mean200_dm)
