from array import array
from tkinter import Y
from matplotlib import pyplot
import json
from turtle import distance
import config
from config import setting,log

import logging,traceback
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
#log = logging.getLogger()

from tqdm      import tqdm
from common    import *

from Trip import read_data_to_trips
from poidata   import load_files,getThemes,getPOIFullNames,getPOIThemes,poi_name_dict
from Bootstrap import inferPOITimes,infer2POIsTimes,get_distance_matrix
from lstm import *

########################
#  TIME LSTM FUNCTIONS
########################
def get_lstm_data_time(pois, userVisits):
  num_pois = len(pois[['poiID']])
  X_input=[]
  y_output=[]
  trip_data = read_data_to_trips(userVisits)
  poiName_df = pd.DataFrame(data=pois, columns=['poiName'])

  log.info("READING TRIP DATA... [%s]", len(trip_data))
  for index in tqdm( range(len(trip_data)) ):
    daylist=trip_data[index]
    #if len( daylist.items ) > 2:
    if len( set([it[1] for it in daylist.items]) ) > 2:
      if True: ### item_times / item_times
        itern_sorted = set(sorted([it[1] for it in daylist.items ] ) )
        #print("LINE {} get_lstm_data_time --> userid: {}, itinerary POIs sorted: {}".format(LINE(),daylist.userid, itern_sorted))

        lastid,lastitem=-1,None
        n=len(daylist.items)
        item_times=[]
        item_ids=[]
        endtime=daylist.items[-1][0]

        for i in range(n):
          if lastid != daylist.items[i][1]:
            item_ids.append(daylist.items[i][1])
            item_times.append(daylist.items[i][0])
          ### loop
          lastitem = daylist.items[i]
          lastid = lastitem[1]

        item_ids.append(-1)
        item_times.append(endtime)
        print("LINE {} IDs:{}, times:{}".format(LINE(),item_ids,item_times))
      ### LSTM model
      for i in range(len(item_times)-3):
        p1,p2,p3 = item_ids[i],item_ids[i+1],item_ids[i+2]
        time1,time2,time3 = item_times[i],item_times[i+1],item_times[i+2]
        if p3>0:
          time4 = item_times[i+3]
          #   onehot_vector(1,num_pois, duration=3.0)
          step1_onehot = onehot_vector(p1, num_pois, duration=time2-time1)
          step2_onehot = onehot_vector(p2, num_pois, duration=time3-time2)
          #step3_onehot = onehot_vector(p3, num_pois, duration= time4-time3)
          print("LINE_66 ", step1_onehot)
          print("LINE_67 ", step2_onehot)
          quit(0)

          step3_onehot = onehot_vector(p3, num_pois)

          X_input.append( [step1_onehot, step2_onehot] )
          y_output.append( step3_onehot )
  print("READTRIP DATA...", len(trip_data))
  #print("LINE... 271")
  #print(X_input,y_output)
  ##for daylist in trip_data:
  return X_input,y_output

def get_lstm_data_time_distance(pois, userVisits):
  num_pois = len(pois[['poiID']])
  X_input=[]
  y_output=[]

  trip_data = read_data_to_trips(userVisits)
  poiName_df = pd.DataFrame(data=pois, columns=['poiName'])
  #onehot_encoder = OneHotEncoder(sparse=False)
  #onehot_encoder.fit(poiName_df)

  log.info("READING TRIP DATA... [%s]", len(trip_data))
  for index in tqdm( range(len(trip_data)) ):
    daylist=trip_data[index]

    #if len( daylist.items ) > 2:
    if len( set([it[1] for it in daylist.items]) ) > 2:
      if True: ### item_times / item_times
        itern_sorted = set(sorted([it[1] for it in daylist.items ] ) )
        #print("LINE {} get_ lstm_data_time --> userid: {}, itinerary POIs sorted: {}".format(LINE(),daylist.userid, itern_sorted))

        lastid,lastitem=-1,None
        n=len(daylist.items)
        item_times=[]
        item_ids=[]
        endtime=daylist.items[-1][0]

        for i in range(n):
          if lastid != daylist.items[i][1]:
            item_ids.append(daylist.items[i][1])
            item_times.append(daylist.items[i][0])
          ### loop
          lastitem = daylist.items[i]
          lastid = lastitem[1]

        item_ids.append(-1)
        item_times.append(endtime)
        log.info("LINE:%d, PRE... IDs:  [%s] , times: %s",LINE(),item_ids,str(item_times) )
        #print("LINE {} PRE... IDs:  {} , times: {}".format(LINE(),item_ids,item_times) )

      log.info("LINE:%d,  len(item_times) -> %d",LINE(), len(item_times))
      ### LSTM model
      for i in range(len(item_times)-3):
        p1,p2,p3 = item_ids[i],item_ids[i+1],item_ids[i+2]
        time1,time2,time3 = item_times[i],item_times[i+1],item_times[i+2]
        if p3>0:
          def vectext(v):
            return ( ",".join( [str(b) for b in v] ) )

          time4 = item_times[i+3]
          km1= (setting['distance_matrix'][(p1,p2)]) / 1000
          km2= (setting['distance_matrix'][(p2,p3)]) / 1000
          hour1= (time2-time1)/3600
          hour2= (time3-time2)/3600

          step1_onehot = onehot_vector(p1, num_pois, distance=km1, duration=hour1)
          step2_onehot = onehot_vector(p2, num_pois, distance=km2, duration=hour2)
          step3_onehot = onehot_vector(p3, num_pois)

          onehot1_text = vectext( step1_onehot[:num_pois] )
          onehot2_text = vectext( step2_onehot[:num_pois] )
          onehot3_text = vectext( step3_onehot[:num_pois] )

          len1,len2= len(step1_onehot),len(step2_onehot)
          len3= len(step3_onehot)

          if  len(step1_onehot)  != len(step2_onehot) :
            log.error("step1_onehot and step2_onehot are not correct size")
            log.error(f"step1_onehot : {step1_onehot}")
            log.error(f"step2_onehot : {step2_onehot}")
            quit(-1)

          if  (len(step1_onehot) - num_pois) != 3:
            log.error("step1_onehot is not correct format")
            log.error(f"num_pois:{num_pois}")
            log.error( "len(step1_onehot):{}".format(len(step1_onehot)))
            print("step1_onehot : \n", step1_onehot)
            quit(-1)

          if (len(step2_onehot)) - (num_pois) != 3:
            log.error("step2_onehot is not correct format")
            log.error(f"num_pois:{num_pois}")
            log.error( "len(step1_onehot):{}".format(len(step2_onehot)))
            log.error("step2_onehot is not correct format")
            log.error("step2_onehot: ", step2_onehot)
            print("step2_onehot : \n", step2_onehot)
            quit(-2)

          print(f"LINE_125: p1:<{onehot1_text}> ({len1}), hour1:{hour1}, km1:{km1}")
          print(f"LINE_125: p2:<{onehot2_text}> ({len2}), hour2:{hour2}, km1:{km2}")
          print(f"LINE_125: p3:<{onehot3_text}> ({len3}),")

          X_input.append( [step1_onehot,step2_onehot] )
          y_output.append( step3_onehot )

  print("READTRIP DATA...", len(trip_data))
  #print("LINE... 271")
  #print(X_input,y_output)
  ##for daylist in trip_data:

  return X_input,y_output

### TRAINING

def lstm_timedistance(pois, userVisits, epoches, dropout, debug=True):
  num_pois = len(pois[['poiID']])

  # WRITE: X_input, y_class, y_reg
  X_input,y_output=[],[]

  ###
  ### READING FROM CSV / POI data / TRAINING
  ###
  X_input,y_output = get_lstm_data_time_distance(pois, userVisits)

  for i in range(len(X_input)):
    print("\n--> i:{}".format(i))
    print("--> X_input[{}][0] ({}) = {}".format(i,len(X_input[i][0]), X_input[i][0]))
    print("--> X_input[{}][1] ({}) = {}".format(i,len(X_input[i][1]), X_input[i][1]))
    print("--> y_output[{}]   ({}) = {}".format(i,len(y_output[i]),   y_output[i]))

    if len( y_output[0]) + 2 != len( X_input[0][0] ):
      log.error("LINE %d X_input:  [%d][%d] ", LINE(), len(X_input[0]),len(X_input[0][0]) )
      log.error("LINE %d y_output: [%d] ", LINE(),len(y_output[0]) )
      quit(0)
  ###
  ### FINSH READING FROM CSV / POI data
  ###

  #--  X_input[0]  : [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  #--  X_input  :  499
  #-- y_output[0]  :  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  #-- y_output  :  499
  ### INPUT to LSTM:  X, y_output, yreq
  if len(X_input) == len(y_output):
    print("LINE {} TRAINING LSTM of {} input data point(s) \n".format(LINE(),len(X_input)))
    print(lstm_model)

    #X_input=[ [X_input[0][0], X_input[0][1] ] ]
    X_input=[ [X_input[0][0] ,X_input[0][1] ] ]
    y_output=[y_output[0] + [9] ] 

    y_reg = np.array( [2] * len(y_output) )

    #LINE_217 --  X_input[0] :[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 13678],
    #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1416 ]]
    #LINE_218 --  X_input  :1
    #LINE_219 -- y_output[0] : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 9, 9]
    #LINE_220 -- y_output  :1

    print( "LINE_217 --  X_input[0]  : ", X_input[0])
    print( "LINE_218 --  X_input  : ", len(X_input))
    print( "LINE_219 -- y_output[0]  : ", y_output[0])
    print( "LINE_220 -- y_output  : ", len(y_output))

    ls_model = lstm_model( X_input, y_output, y_reg, _epochs=epoches, _dropout=dropout, _verbose_level=3)
    print("LINE {}, LSTM model: {}".format(LINE(),str(ls_model)))
    return ls_model

  else:
    log.error("X_input and y_class must be equal!")
    log.error("FILE {}, LINE {}".format(sys.argv[0],LINE()))
    quit(0)

### EVALUATE / PREDICTION
def get_first_event(user_seq_visits):
  p1_starttime = user_seq_visits['dateTaken'].min()
  first_poi_df = user_seq_visits[ user_seq_visits['dateTaken']==p1_starttime ]
  first_poiid = first_poi_df['poiID'].max()
  first_poi_df = user_seq_visits[ user_seq_visits['poiID']==first_poiid ]
  minTime=user_seq_visits['dateTaken'].min()
  maxTime=user_seq_visits['dateTaken'].max()
  return first_poiid,minTime,maxTime,first_poi_df

def get_second_event(user_seq_visits):
  first_poiid,minFirstTime,maxFirstTime,first_poi_df = get_first_event(user_seq_visits)
  not_first_poi_df = user_seq_visits[ user_seq_visits['dateTaken'] > maxFirstTime]
  if not not_first_poi_df.empty:
    return get_first_event(not_first_poi_df)
  else:
    return None

def get_first_POI_time(userid,seqid,user_seq_visits):
  user_seq_visits.sort_values(by=['dateTaken'])
  poiids = sorted(user_seq_visits['poiID'].unique())

  (first_poiid,p1_starttime,p1_endtime,first_poi_df) = get_first_event(user_seq_visits)

  second_poi_df = user_seq_visits[ user_seq_visits['dateTaken'] > p1_endtime ]

  if not second_poi_df.empty:
    #print("-------------------- first_poi_df:")
    #print(first_poi_df)
    #print("--------------------")
    p1_starttime = first_poi_df['dateTaken'].min()
    p1_endtime = second_poi_df['dateTaken'].min()
    #print("LINE {}, p1 : <poiID: {}> [ {} .. {} ] -- {} seconds".format(LINE(), first_poiid ,p1_starttime, p1_endtime, p1_endtime-p1_starttime))
  return first_poiid , (p1_endtime-p1_starttime)

def get_second_POI_time (userid,seqid,user_seq_visits):
  user_seq_visits.sort_values(by=['dateTaken'])
  poiids = sorted(user_seq_visits['poiID'].unique())
  if len(poiids) <= 1: return None,None

  #print("LINE {}, get_second_POI_time(...)".format(LINE()))
  #print("LINE {}, poiids : {}".format(LINE(),poiids))
  #print("LINE {}, user_seq_visits : \n{}".format(LINE(),user_seq_visits))

  p1_starttime = user_seq_visits['dateTaken'].min()

  first_poi_df = user_seq_visits[ user_seq_visits['dateTaken']==p1_starttime ]
  poiid1 = first_poi_df['poiID'].unique()[0]
  first_poi_df = user_seq_visits[ user_seq_visits['poiID']==poiid1 ]
  #print("LINE {}, -- get_second_POI_time ... first_poi_df :\n{}".format(LINE(),first_poi_df))

  not_first_poi_df = user_seq_visits[ user_seq_visits['poiID'] != poiid1]
  not_first_poi_df.sort_values(by=['dateTaken'])
  
  second_starttime = not_first_poi_df['dateTaken'].min()
  second_poi_df = not_first_poi_df[ not_first_poi_df['dateTaken'] == second_starttime ]
  second_poiid = second_poi_df['poiID'].min()
  second_poi_df = user_seq_visits[user_seq_visits['poiID'] == second_poiid]
  second_endtime = second_poi_df['dateTaken'].max()
  #print("LINE {}, p2 : <poiID: {}> [ {} .. {} ] -- {} seconds".format(LINE(), second_poiid, second_starttime, second_endtime, second_endtime-second_starttime))

  return second_poiid , (second_endtime-second_starttime)

def summerize_prediction(user_seq_visits,userid,seqid, predseq,durations):
  ### HISTORY TRIJECTORY
  trajectory = list(user_seq_visits['poiID'].unique())

  get_first_POI_time(userid,seqid,user_seq_visits)
  sum_durations = sum(durations)

  #print("    sequence:  ", sequence)
  #print("    durations: ", durations)
  #print("    durations: sum (hrs) ", sum(durations))

  print("---------------------")
  print("LINE {}    Trajectory: {}".format(LINE(), str(trajectory)))
  print("LINE {}    Uniq IDs: {}".format(LINE(), user_seq_visits['poiID'].unique()))
  timeDelta = user_seq_visits['dateTaken'].max() - user_seq_visits['dateTaken'].min()
  print("LINE {0:.3g}    time delta : {1:3g}s -> {2:.3g}hrs".format(LINE(),timeDelta,(timeDelta/60/60)))
  #print("\n [user_seq_visits] table:")
  #print(user_seq_visits)

  print("---------------------")
  print("LINE {}    PREDICTED SEQ: {}".format(LINE(), str(predseq)) )
  print("LINE {}      duration: {} ".format(LINE(), str(durations)) )
  print("LINE {}      sum:{}".format(LINE(), sum_durations) )

  jsonel = dict()
            
  trajectory_int =  [ int(el) for el in trajectory]
  jsonel['trajectory'] = (trajectory_int)

  print("LINE {}  jsonel:{}".format(LINE(), jsonel) )

  predseq_int =  [ int(el) for el in predseq]
  jsonel['predicted_sequence'] = (predseq_int)

  print("LINE {}  jsonel:{}".format(LINE(), jsonel) )

  durations_float =  [ float(el) for el in durations]
  jsonel['predicted_duration'] = durations_float

  print("LINE {}  jsonel:{}".format(LINE(), jsonel) )
  return jsonel

def lstm_timedistance_predict(lsmodel, visits):
  log.info("LINE %d, lstm_ timedistance_ predict(lsmodel, visits)", LINE())
  "SCANNING the visits-dataframe as 'TESTING' data"
  log.info("LINE %d, EVALUATE LSTM -- lstm_time_predict(lsmodel, visits)", LINE())

  data = []

  for userid in sorted(visits['userID'].unique()):
    user_visits = visits[ visits['userID'] == userid ]
    print("==>  USERID : ", userid)

    for seqid in sorted(user_visits['seqID'].unique()):
      #print("===>   SEQID : ", seqid)
      user_seq_visits = user_visits[ user_visits['seqID'] == seqid ]

      p1,p1time = get_first_POI_time(userid,seqid,user_seq_visits)
      p2,p2time = get_second_POI_time(userid,seqid,user_seq_visits)
      if p1time and p2time:
        user_seq_visits_p3s = user_seq_visits[ (user_seq_visits["poiID"] != p1) ]
        user_seq_visits_p3s = user_seq_visits_p3s[ (user_seq_visits_p3s["poiID"] != p2) ]

        if not user_seq_visits_p3s.empty:
          min_timesstamp = user_seq_visits['dateTaken'].min()
          max_timesstamp = user_seq_visits['dateTaken'].max()
          tryject_duration=max_timesstamp-min_timesstamp+1

          ### MINIMUM TIME IS 8 HRS
          if tryject_duration < 8 * 60 * 60:
            tryject_duration =8 * 60 * 60

          print("\n  ==================================================")
          log.debug("line %d LSTM_TIME_PREDICT ... userid:%8s, seqid:%2d", LINE(), userid, seqid)

          ### PREDICT POIS of some duration 
          # (predseq,durations) = lstm_timedistance_predict_sequence(lsmodel, tryject_duration, p1, p1time, p2, p2time)

          (predseq,durations) = lstm_timedistance_predict_sequence(lsmodel, tryject_duration, 0,0, p1, p1time)
          jsonel = summerize_prediction(user_seq_visits,userid,seqid, predseq,durations)
          data.append(jsonel)
          print("LINE {}  data:{}".format(LINE(), data) )

  #log.info("LINE %d -- JSON DATA: %d", LINE(), len(data) )
  log.info("LINE %d -- DATA: %s ", LINE(), data)
  return data

def distance(pois,p1,p2):
  print(pois)
  quit(0)
  pois['']
  from Bootstrap import get_distance
  distance = get_distance([i_lat,i_lng], [j_lat,j_lng], "euclidean")

def lstm_timedistance_predict_sequence(lsmodel, total_duration, p1, p1time, p2, p2time):
  log.info("  LINE %d -- lstm_timedistance_predict_sequence( <lsmodel>, total_duration=%d, p1=%d, p1time=%d, p2=%d, p2time=%d...",
           LINE(), total_duration, p1,p1time, p2, p2time)
  _pois = setting['POIs']
  _City = setting['CITY']
  _Epoches = setting['EPOCHES']
  _Dropout = setting['DROPOUT']
  
  num_pois = _pois['poiID'].max()

  sequence=[p1, p2]
  print("  line {}  LEADING sequence: {}".format(LINE(), sequence))
  durations=[p1time/3600, p2time/3600]
  print("  line {}  LEADING duration: {}".format(LINE(), durations))

  totaltime = (p1time+p2time)/ 3600  # convert to hours
  total_duration_hour = total_duration / 60 / 60

  log.info("LINE %d, lstm_timedistance_predict_sequence(lsmodel, total_duration:%d, ...) ", LINE(), total_duration)
  log.info("LINE %d,  --> total_duration_seconds : %d", LINE(), total_duration)
  log.info("LINE %d,  --> total_duration_hour    : %f", LINE(), total_duration_hour)
  total_distance=0
  while totaltime < total_duration_hour:
    from Bootstrap import get_distance
    print("LINE {} p1: {}, p2: {}".format(LINE, p1, p2))
    p1_frame = _pois[ _pois['poiID']==p1+1 ]
    p2_frame = _pois[ _pois['poiID']==p2+1 ]
    p1_lat = p1_frame['lat'].min()
    p1_lon = p1_frame['long'].min()
    p2_lat = p2_frame['lat'].min()
    p2_lon = p2_frame['long'].min()

    print("LINE {} \n{}\n{}\n".format(LINE(), p1_frame, p2_frame))

    dist_1_2 = get_distance([float(p1_lat),float(p1_lon)], [float(p2_lat),float(p2_lon)])
    print("LINE {}, dist_1_2 = get_distance([float({}),float({})], [float({}),float({})])".format(LINE(),p1_lat,p1_lon,p2_lat,p2_lon,))

    duration1 = durations[-2]
    duration2 = durations[-1]

    print("LINE {}, dist_1_2 : {}".format(LINE(),dist_1_2))
    print("LINE {}, duration1 = get_distance([float({}),float({})], [float({}),float({})])"\
      .format(LINE(),p1_lat,p1_lon,p2_lat,p2_lon,))

    onehot_v0 = onehot_vector(p1, num_pois)
    onehot_v1 = onehot_vector(p1, num_pois, duration=duration1, distance=dist_1_2)
    onehot_v2 = onehot_vector(p2, num_pois, duration=duration2, distance=dist_1_2)
    print("LINE {} distance : {}".format(LINE(), dist_1_2))

    print(f"p1:{p1} onehot_v1 -> ", onehot_v1)
    print(f"p2:{p2} onehot_v2 -> ", onehot_v2)

    # expected shape=(None, 2, 32), found shape=(None, 2, 31)

    #onehot_v1 = onehot_vector(poiid1, num_pois, duration=duration1 )
    #onehot_v2 = onehot_vector(poiid2, num_pois, duration=duration2 )
    # expected shape=(None, 2, 32), found shape=(None, 2, 31)
    # expected shape=(None, 2, 32), found shape=(None, 2, 31)

    #onehot_v1 = onehot_vector(poiid1, num_pois)
    #onehot_v2 = onehot_vector(poiid2, num_pois)
    # expected shape=(None, 2, 32), found shape=(None, 2, 31)
    # expected shape=(None, 2, 32), found shape=(None, 2, 31)

    print("onehot_vector(poiid2={}, num_pois={}, distance={})". format(p2,num_pois,dist_1_2))

    print("\nLINE {}  onehot_v0 : {}".format(LINE(), onehot_v0))
    print(  "LINE {}  onehot_v1 : {} ({}),\n\t duration:{}".format(LINE(), len(onehot_v1), onehot_v1, duration1))
    print(  "LINE {}  onehot_v2 : {} ({}),\n\t duration:{}".format(LINE(), len(onehot_v2), onehot_v2, duration2))

    X=[ [onehot_v1, onehot_v2] ]

    print("LINE {},  y_durations, y_poiids = lsmodel.predict(X) ".format(LINE()))

    #Input 0 is incompatible with layer model:
    #  expected shape=(None, 2, 31), found shape=(None, 2, 32)
    log.debug("\nline %d X[0][0]: %s", LINE(), str(X[0][0]) )
    log.debug("\nline %d X[0][1]: %s", LINE(), str(X[0][1]) )
    y_durations, y_poiids = lsmodel.predict(X)

    y_pred_duration = y_durations[0][0][0]
    y_poiids = y_poiids[0][0]

    print("LINE {}  y_poiids -> {}".format(LINE(), y_poiids))
    for i in range(num_pois):
      poiid = i+1
      print("  num_pois->{}  y_poiids[ {} ] -> {}".format( num_pois, poiid, y_poiids[i] ) )

    print(y_poiids)
    print("  y_pred_duration -> ", y_pred_duration)

    ### ARGMAX of all y_poiids
    ### SET VISIT POIs to -1
    total_distance = total_distance + dist_1_2
    y_poiids[0] = -1

    for poiid in sequence: y_poiids[poiid] = -1

    if max(y_poiids) > 0:
      y_pred_poiid = np.argmax(y_poiids)
      sequence.append(y_pred_poiid)
      durations.append(y_pred_duration)

      #log.debug("LINE %d ADD TO SEQUENCE... id: %2d, duration: %s", LINE(), y_pred_poiid, str(y_pred_duration))

      totaltime += y_pred_duration

      #print(" | LINE {}, y_pred_poiid    : {}".format(LINE(), y_pred_poiid) )
      #print(" | LINE {}, y_pred_duration : {} (hrs) ~~ {} mins".format(LINE(), y_pred_duration, y_pred_duration*60 ))
      #print("  ===-------------------- \n".format(LINE()))
    else:
      #print("  LINE {} NOTHING TO ADD TO SEQUENCE... y_poiids : {}".format(LINE(), str(y_pred_poiid)))
      #print("  LINE {} SEQUENCE ... {}".format(LINE(), str(sequence)))
      totaltime = 888*60*60
    #print("  LINE:{} Sequece : {}".format(LINE(), sequence))
    #print("  LINE:{} Durations: {}\n".format(LINE(), durations ))

  print("  line:{} FINAL SEQUENCE: {}".format(LINE(), sequence))
  print("  line:{} FINAL DURATIONS: {}".format(LINE(), durations ))
  return( sequence,durations )

def XXX_lstm_eval_poiid(lmodel, X):
  print("\n\n### ==========================\n-- lstm_eval_poiid(...)")
  ### predict
  y_durations, y_poiids = lmodel.predict(X)

  if (len(X)==1):
    print("LINE {}, y_poiids : {}".format( LINE(), y_poiids.tolist()) )
    [amax] = [ amax_arr[0] for amax_arr in y_poiids.argmax(axis=2) ]
    print("\n----- ----- ----- -----\n-- LINE {} PREDICTION: ".format(LINE()))
    #print("==> L {} \t y_pred_poiid  :   \t {}".format( LINE(), amax ))
    #print("==>     \t y_poiids (size) : ", len(y_poiids) )
    x_1 = X[0][0]
    x_2 = X[0][1]
    x1_ID,_,x1_secs = onehot_poiID(x_1)
    x2_ID,_,x2_secs = onehot_poiID(x_2)
    duration=y_durations[0][0][0] 
    print( "LINE {}: x1_ID: {}".format(LINE(), x1_ID) )   
    print( "LINE {}: x2_ID: {}".format(LINE(), x2_ID) )
    print( "LINE {}:  y_ID: {}, duration/dist: {}\n".format(LINE(), amax, duration) )
    #print( y_durations[0][0][0] )
    #print( "\nLINE {}: y_poiids: {}".format(LINE(), y_poiids[0][0]) )

  else:
    for t in range(len(y_poiids)):
      print( "\nLINE {} y_poiids[{}]: \n{}".format(LINE(), t, y_poiids[t][0]) )

  #print("==> L{} \t y_durations: size:{} \t {} (mins) ".format( LINE(),len(y_durations) , [ (60*t[0][0]) for t in y_durations] ) )
  #print("==> L{} \t y_durations:--> X_input size:{} \t {} (mins) ".format( LINE(),len(y_durations) , [ (60*t[0][0])  ] ) )
  #for t in range(len(y_durations)):
  #  print( "LINE {}: y_durations[{}]: {:0.0f} sec".format(LINE(), t, 60 * (y_durations[t][0][0]) ) )
  print("== lstm_eval_poiid(...)")
  return y_durations, y_poiids

########################
#  DISTANCE LSTM FUNCTIONS
########################

def lstm_distance(city, epoches, dropout, debug=True):
  quit(0)

def lstm_eval_distance(lmodel, X):
  print("\n\n-- lstm_eval_poiid(...)")
  quit(0)
  ### predict
  y_pred_distance, y_poiids = lmodel.predict(X)

  if (len(X)==1):
    print("\n y_poiids : ", str(y_poiids) )
    [amax] = [ amax_arr[0] for amax_arr in y_poiids.argmax(axis=2) ]
    print("\n ----- ----- ----- -----\n LINE386 -- PREDICTION: ")
    #print("==> L {} \t y_pred_poiid  :   \t {}".format( LINE(), amax ))
    #print("==>     \t y_poiids (size) : ", len(y_poiids) )

    x_1 = X[0][0]
    x_2 = X[0][1]
    x1_ID,_,x1_secs = onehot_poiID(x_1)
    x2_ID,_,x2_secs = onehot_poiID(x_2)

    print("y_pred_distance([0][0] : ", y_pred_distance.flatten())
    if len( y_pred_distance.flatten() ) != 1:
      log.error("LINE %d, y_pred_distance contains > 1 value!", LINE())
    distance=y_pred_distance.flatten()[0] 

    print( "--LINE {}: x1_ID: {}".format(LINE(), x1_ID) )
    print( "--LINE {}: x2_ID: {}".format(LINE(), x2_ID) )
    print( "--LINE {}:  y_ID: {}, y_distance : {}\n".format(LINE(), amax, 1000*distance) )

  else:
    for t in range(len(y_poiids)):
      print( "\nLINE {} y_poiids[{}]: \n{}".format(LINE(), t, y_poiids[t][0]) )

  #print("==> L{} \t y_durations: size:{} \t {} (mins) ".format( LINE(),len(y_durations) , [ (60*t[0][0]) for t in y_durations] ) )
  #print("==> L{} \t y_durations:--> X_input size:{} \t {} (mins) ".format( LINE(),len(y_durations) , [ (60*t[0][0])  ] ) )
  #for t in range(len(y_durations)):
  #  print( "LINE {}: y_durations[{}]: {:0.0f} sec".format(LINE(), t, 60 * (y_durations[t][0][0]) ) )
  print("== lstm_eval_poiid(...)")

  return distance, y_poiids

def main():
  ## DEBUG
  ## city='Pert'
  ## (pois, userVisits, testVisits, _) = load_files(city, fold=5, subset=1, DEBUG=1)
  ## model = lstm_time(pois, userVisits, testVisits,  epoches=100, dropout=0.2)
  ## quit(0)
  ## DEBUG

  ## default params
  city=''
  debug=1
  _Epoches = 200
  _Dropout = 0.25

  if len(sys.argv) <= 3:
    print ("SYNTAX: {} <city> <EPOCHES> <DROP OUT RATE>".format(sys.argv[0]))
    print ("EXAMPLEX: {} Osak {} {}".format(sys.argv[0], _Epoches, _Dropout))
    quit(0)
  else:
    city=sys.argv[1]
    if city not in ['Buda','Delh','Edin','Glas','Osak','Pert','Toro']:
      log.error("CITY IS NOT FOUND : [ %s ]", city)
      quit(-1)

    if len(sys.argv) > 2:
      _Epoches = int(sys.argv[2])
      _Dropout = float(sys.argv[3])
      log.info("RUNTIME: %s CITY:[%s] [%d] [%f]", sys.argv[0],city, _Epoches, _Dropout)

  setting['EPOCHES'] = _Epoches
  setting['DROPOUT'] = _Dropout

  #METHOD="LSTM_PHOTO" ## "LSTM_TIME" / "LSTM_DISTANCE" 
  #METHOD="LSTM_DISTANCE"
  #METHOD="LSTM_TIME" 
  METHOD="LSTM_TIME_DISTANCE" 
  log.debug("MODE : [%s]", METHOD)

  FOLDS=5
  AllFolds_fscore_list = []
  AllFolds_precision_list = []
  AllFolds_recall_list = []

  for FOLD in range(1,FOLDS+1):
    ### CROSS VALIDATION
    setting["FOLD"]=FOLD
    setting["FOLDS_CROSS_VALIDATION"]=FOLDS

    (pois, userVisits, testVisits, _) = load_files(city, fold=FOLDS, subset=FOLD, DEBUG=1)
    from Bootstrap import get_distance_matrix
    dist_mat = get_distance_matrix(pois)
    setting['distance_matrix'] = dist_mat
    setting['POIs'] = pois

    if METHOD=="LSTM_TIME_DISTANCE": ### USING REAL DATA -- PREDICT DISTANCE
      print("lstm_timedistance")

      model = lstm_timedistance(pois, userVisits, _Epoches, _Dropout, debug=True)

      ### TRAINING SET
      results = lstm_timedistance_predict(model, userVisits)
      print("LINE686")
      ### TESTING DATA SET
      #results = lstm_timedistance_predict(model, testVisits)
      fscore_list,precision_list,recall_list=[],[],[]
      for result in results:
        #print()
        #print(result)
        
        trajectory = result['trajectory']
        if (type(trajectory) == 'str' ): trajectory = eval(trajectory)
        
        predicted = result['predicted_sequence']
        if (type(predicted) == 'str' ): predicted = eval(predicted)
        #print("predicted : ", type(predicted))

        duration_list = result['predicted_duration']
        if (type(duration_list) == 'str' ): duration_list = eval(duration_list)

        #print(" trajectory           => ", " |".join(trajectory))
        if predicted[0] == 0:  predicted=predicted[1:]
        if trajectory[0] == 0:  trajectory=trajectory[1:]
        print("\n ==> trajectory           -> ", (trajectory))
        print(" ==> predicted           -> ", (predicted))

        star_predicted = []
        for i in range(len(predicted)):
          str=""
          p = int(predicted[i])
          if p in trajectory:
            str = "{}*".format(p)
          else:
            str = "{}".format(p)
          star_predicted.append(str)

        print("   predicted_sequence -> ", star_predicted)
        print("   predicted_duration -> ", result['predicted_duration'])

        from common import f1_scores
        precision,recall,fscore = f1_scores(trajectory, predicted)
        fscore_list.append(fscore)
        precision_list.append(precision)
        recall_list.append(recall)
        print("LINE {}, f1:{}\t precision:{}\t recall:{}".format(LINE(),fscore,precision,recall))

      #plotFile="Prec-Rec-Curve_{}-e{}-d{}.png".format(city,_Epoches,_Dropout)
      #pyplot.plot( recall_list, precision_list, marker='.', label='Logistic')
      #pyplot.savefig(plotFile, orientation='landscape', dpi=500, bbox_inches='tight')
      #log.info("wrote Prec.Recall.Curve: %s", plotFile)

      for s in fscore_list: AllFolds_fscore_list.append(s) 
      for s in precision_list: AllFolds_precision_list.append(s) 
      for s in recall_list: AllFolds_recall_list.append(s) 

      ### SUMMARY fscore_list,precision_list,recall_list
      header="<CITY:{}> <EPOCHES:{}> <DROPOUT:{}>".format(city,_Epoches,_Dropout)
      f1score_pct = 100*np.mean(fscore_list)
      f1score_err = 100*np.std(fscore_list)
      precision_pct = 100*np.mean(precision_list)
      precision_err = 100*np.std(precision_list)
      recall_pct = 100*np.mean(recall_list)
      recall_err = 100*np.std(recall_list)

      print(f"{header}\t FOLD:{FOLD} F1-scores:         mean: {f1score_pct} %,   stderr: {f1score_err} %")
      print(f"{header}\t FOLD:{FOLD} Precision-scores:  mean: {precision_pct} %, stderr: {precision_err} %")
      print(f"{header}\t FOLD:{FOLD} Recall-scores:     mean: {recall_pct} %,    stderr: {recall_err} %")

  ### calculate cross validation
  f1score_pct = 100*np.mean(AllFolds_fscore_list)
  f1score_err = 100*np.std(AllFolds_fscore_list)
  precision_pct = 100*np.mean(AllFolds_precision_list)
  precision_err = 100*np.std(AllFolds_precision_list)
  recall_pct = 100*np.mean(AllFolds_recall_list)
  recall_err = 100*np.std(AllFolds_recall_list)
  print(f"{header}\t {FOLDS}-CROSS_FOLD F1-scores:         mean: {f1score_pct} %,   stderr: {f1score_err} %")
  print(f"{header}\t {FOLDS}-CROSS_FOLD Precision-scores:  mean: {precision_pct} %, stderr: {precision_err} %")
  print(f"{header}\t {FOLDS}-CROSS_FOLD Recall-scores:     mean: {recall_pct} %,    stderr: {recall_err} %")

if __name__ == '__main__':
  main()
