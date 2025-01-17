

import sys
import logging
import pandas as pd
import numpy as np
from config import setting,log


def corpus_text(text):
  ### only CAPTITAL LETTER, no a/e/i/o/u/ no symbol
  text = text.upper()  \
    .replace(",","")   \
    .replace(",","")   \
    .replace("[","")   \
    .replace("]","")   \
    .replace("(","")   \
    .replace(")","")   \
    .replace("_","_")   \
    .replace("-","_")
  return text

def poi_name_dict(pois):
  p2n=dict()
  n2p=dict()
  poiNames = getPOINames(pois)
  poiFullNames = getPOIFullNames(pois)
  for i in poiNames:
    ## i <-> poiNames[i]
    name = poiNames[i]
    p2n[i]    = name
    n2p[name] = i
  return (p2n,n2p)

def load_files(city, fold=1, subset=1, DEBUG=0):
  def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

  IN_COLAB = 'google.colab' in sys.modules
  logging.info("LINE 37, IN_COLAB: {}".format(IN_COLAB))

  Themes = ["Amusement", "Beach", "Cultural", "Shopping", "Sport", "Structure"]
  POI_file=''

  if IN_COLAB:
    drive_path="/content"
    POI_file         = drive_path + "/Data/POI-"         + city+".csv"
    userVisits_file  = drive_path + "/Data/userVisits-"  + city+"-allPOI.csv"
    #costProfCat_file = drive_path + "/Data/costProfCat-" + city+"POI-all.csv"
  else:
    POI_file = "Data/POI-{}.csv".format(city)
    userVisits_file = "Data/userVisits-{}-allPOI.csv".format(city)
    #costProfCat_file = "Data/costProfCat-"+city+"POI-all.csv"

  ### ADD IN LONG NAMES
  print("L53: reading poi file: ", POI_file)
  pois = pd.read_csv(POI_file, sep=';', dtype={'poiID':int, 'poiName':str, 'lat':float, 'long':float, 'theme':str} )
  pois['poiLongName'] = pois['poiName']
  pois.style.set_properties(**{'text-align': 'left'})
  pois['poiName'] = pois['poiLongName'].apply(lambda x: corpus_text(x))

  ### USE SHORT NAMES
  ### pois['poiName'] = pois['poiLongName'].apply(lambda x: corpus_text(x))
  ### pois.drop(columns=['poiName2'], axis=1)
  # costProfCat= pd.read_csv(costProfCat_file, sep=';', dtype={'photoID':int, 'userID':str, 'dateTaken':int, 'poiID':int, 'poiTheme':str, 'poiFreq':int, 'seqID':int} )
  userVisits = pd.read_csv(userVisits_file, sep=';', dtype={'photoID':int, 'userID':str, 'dateTaken':int, 'poiID':int, 'poiTheme':str, 'poiFreq':int, 'seqID':int} )

  ### remove seqid with only a few photos
  remove_index_list,remove_list,index_list=[],[],[]
  all_seqid_set = userVisits['seqID'].unique()
  finish_time=dict()

  ### bootstrapping
  from Bootstrap import inferPOITimes
  boottable=userVisits.copy()

  ### shrink userVisits to only 3 or more
  for seqid in all_seqid_set:
    userVisits_seqid = userVisits[ userVisits['seqID'] == seqid ]
    poiids = userVisits_seqid['poiID'].unique()
    #print(poiids)
    if len(poiids) < 4:
      t = userVisits[ userVisits['seqID'] != seqid ]
      userVisits = t
    else:
      # record finish time
      finish_time[seqid] = userVisits_seqid['dateTaken'].max()

  n=len(finish_time)
  all_finish_times = (sorted( finish_time.values()))
  lastindex=int(n * 80 / 100)
  max_training_time = max( all_finish_times[ 0 : lastindex ] )

  ### bootstrapping
  #from Bootstrap import inferPOITimes,infer2POIsTimes
  #boottable=userVisits.copy()
  boottable=boottable[ boottable['dateTaken'] <= max_training_time]
  #print(boottable)
  boot_times = inferPOITimes(pois,boottable)
  setting['bootstrap_time'] = boot_times

  drop_seqids,keep_seqids=[],[]

  for seqid in all_seqid_set:
    userVisits_seqid_table = userVisits[ userVisits['seqID'] == seqid ]
    if userVisits_seqid_table.empty:
      drop_seqids.append(seqid)
    else:
      #print("==> userVisits_seqid_table : \n", userVisits_seqid_table)
      if userVisits_seqid_table['dateTaken'].max() <= max_training_time:
        keep_seqids.append(seqid)
      else:
        drop_seqids.append(seqid)

  userVisits_training = filter_rows_by_values(userVisits, "seqID", drop_seqids)
  userVisits_testing = filter_rows_by_values(userVisits, "seqID", keep_seqids)
  #print("{} userVisits.shape: {}".format(city,userVisits.shape))
  #print("{} userVisits_training.shape: {}".format(city, userVisits_training.shape))
  #print("{} userVisits_testing.shape:  {}".format(city, userVisits_training.shape))
  #logging.info("### poidata.py ### load_files LOADED")

  if "CITY" in setting: setting["CITY"]=city
  ## CROSS VALIDATION SET ?
  '''
  if "FOLDS_CROSS_VALIDATION" in setting and setting["FOLDS_CROSS_VALIDATION"]:
    logging.info("poidata: LINE 85, using cross validation.." )
    quit(-1)
    setting_FOLD = setting["FOLD"]
    setting_FOLDS = setting["FOLDS_CROSS_VALIDATION"]
    logging.info("  Crossing Valudation: %d / %d", setting["FOLD"], setting["FOLDS_CROSS_VALIDATION"] )
    users = ( userVisits["userID"].unique() )
    remove_users = []
    keep_users = []
    for i in range(len(users)):
      if setting_FOLD == i % setting_FOLDS:
        remove_users.append(users[i])
      else:
        keep_users.append(users[i])
    logging.debug("poidata.py:   dropped %d rows, from %d-FOLDS_CROSS_VALIDATION", len(remove_users), setting["FOLDS_CROSS_VALIDATION"])
    logging.debug("poidata.py: userVisits.shape : %s", str(userVisits.shape))
    testVisits = userVisits.copy()
    for ruser in keep_users:
      remove_index = testVisits[ testVisits['userID'] == ruser ].index
      testVisits.drop(remove_index, inplace = True)
    logging.debug("testVisits.shape : %s", str(testVisits.shape))
    for ruser in remove_users:
      remove_index = userVisits[ userVisits['userID'] == ruser ].index
      userVisits.drop(remove_index, inplace = True)
    logging.debug("userVisits.shape : %s", str(userVisits.shape))
    print("RETURN: {} pois       : {}".format(city,pois.shape))
    print("RETURN: {} userVisits : {}".format(city,userVisits_training.shape))
    print("RETURN: {} testVisits : {}".format(city,userVisits_testing.shape))
  '''
  #return(pois, userVisits_training, userVisits_testing, costProfCat)
  return(pois, userVisits_training, userVisits_testing, boot_times)

def getThemes(pois):
  sorted_themes = pois.sort_values("theme")
  return sorted_themes['theme'].unique()

def getAllUsers(userVisits):
  users = userVisits[['userID']].drop_duplicates(subset=['userID'])
  return users

def getUser2User(users,pois):
  user2userDF = users.join(users, lsuffix='_a', rsuffix='_b')
  print(user2userDF.head())
  jr = user2userDF.shape[1]

  if False:
    d = []
    for i in range(jr):
      u1 = user2userDF['userID_a'][i]
      u2 = user2userDF['userID_b'][i]

      if u1 == u2:
        d.append(i)
        print("3 user2userDF['userID_a'][i] : ", user2userDF['userID_a'][i])
    print("### drop : ", d)
    user2userDF = user2userDF.drop(d)

  data = np.array(user2userDF)
  data2 = []
  for i in data.tolist():
    for p in pois["poiID"]:
      data2.append(  i + [p] )
      #print("  poiId : ", p)
  #print(data2)
  data3 = pd.DataFrame(columns=['userID1','userID2','poiID'],data=data2)
  return data3

def getPOIIDs(df):
  poiIDs=[]
  for i in df["poiID"]:
    poiIDs.append(i)
  return poiIDs

def getPOIThemes(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    poiIDs[id] = row['theme']
  return poiIDs

def getPOINames(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    name = row['poiName']
    poiIDs[id] = name
  return poiIDs

def getPOIFullNames(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    poiIDs[id] = row['poiLongName']
  return poiIDs

def getPOILongNames(df):
  poiIDs=dict()
  for index, row in df.iterrows():
    id = row['poiID']
    name = row['poiLongName']
    poiIDs[id] = name
  return poiIDs

def unittest():
  import numpy as np
  for city in ["Buda", "Delh", "Edin", "Glas", "Osak", "Pert", "Toro", "Vien" ]:
    pois, userVisits, testVisits, costProfCat = load_files(city, fold=1, subset=1, DEBUG=1)
    #seqid=70
    #print( testVisits['userID'] )
    #print( testVisits['seqID'] )
    #print( testVisits['seqID'] == seqid )
    #print( testVisits[ testVisits['seqID'] == seqid ] )

    print("---------------------\ncity: {}".format(city))
    print("train unique users: ", userVisits['userID'].unique().shape)
    print("train        photos: ", userVisits['userID'].shape)
    #print("train avg    photos: ", (userVisits['userID'].shape[0] / userVisits['userID'].unique().shape[0]) )
    print("testing unique users: ", testVisits['userID'].unique().shape)
    print("testing        photos: ", testVisits['userID'].shape)

    num_pois_vec = []
    num_photos_vec = []
    for seqid in userVisits['seqID'].unique():
      #print("  seqid : ", seqid)
      seqid_table = userVisits[ userVisits['seqID']==seqid ]
      #print(seqid_table)
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()

      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    print(f"{city} num_tryj :              {len(num_pois_vec)}")
    print(f"{city} num_test_tryj :         {len(num_pois_vec)}")
    print(f"{city} num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    print(f"{city} num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )

    num_pois_vec = []
    num_photos_vec = []
    for seqid in testVisits['seqID'].unique():
      seqid_table = userVisits[ userVisits['seqID']==seqid ]
      numpois = len( seqid_table['poiID'].unique())
      numcheckins = seqid_table['poiID'].count()
      num_pois_vec.append( numpois )
      num_photos_vec.append( numcheckins )

    print(f"{city} (test) num_tryj :              {len(num_pois_vec)}")
    print(f"{city} (test) num_test_tryj :         {len(num_pois_vec)}")
    #print(f"{city} (test) num_pois_vec / tryj :   {np.mean(num_pois_vec)}")
    #print(f"{city} (test) num_photos_vec / tryj : {np.mean(num_photos_vec) / np.mean(num_pois_vec)}" )

if __name__ == "__main__":
  unittest()
