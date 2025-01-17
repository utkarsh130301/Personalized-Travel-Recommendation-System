import argparse
import pandas as pd
import numpy as np
from subseq.subseq import Subseq
# Succinct BWT library
from common import f1_scores, LINE, log

from poidata import (
    load_files,getThemes,getPOIFullNames,getPOIThemes,poi_name_dict)

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs,
    MultiLabelClassificationModel, MultiLabelClassificationArgs)

def calc_time_intervals(pois, userVisits):
    times=dict()
    # init times[ poiid ] array
    for poi in pois['poiID'].unique():
       times[poi] = []
    # record all time intervals
    for seqID in sorted(userVisits['seqID'].unique()):
       print("  seqid: ", seqID)
       seqtable = userVisits[userVisits['seqID']==seqID]
       print(seqtable)
       for poi in sorted(seqtable['poiID'].unique()):
           poi_seqtable= seqtable[ seqtable['poiID'] == poi ]
           #print(f"seqid: {seqID}, poi: {poi}, \n{poi_seqtable.shape} ")
           if poi_seqtable.shape[0] > 1:
               timediff= poi_seqtable['dateTaken'].max() - poi_seqtable['dateTaken'].min()
               print(f"seqid: {seqID}, poi: {poi}, time: {timediff}")
               times[poi].append(timediff)
    print(times)
    time_intervals=dict()
    for poi in pois['poiID'].unique():
       print(f" times[ {poi} ] : {times[poi]}")
       if len(times[poi]) > 0:
           mean_time = np.mean(times[poi])
           time_intervals[poi] = mean_time
       else:
           time_intervals[poi] = 5 * 60 ### default : 5 mins
    print("INTERVALS : ", time_intervals)
    return time_intervals

def run_city( city):
    from Bootstrap import inferPOITimes2
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', '-c', type=str, required=True)
    args = parser.parse_args()
    print('### ARGS:       {}'.format(str(args)))

    # read in from  spmf.sh /
    ### for city in ['Buda','Delh','Edin','Glas','Osak','Pert','Toro']
    (pois, userVisits, testVisits, costProfCat) = load_files( args.city )

    times = calc_time_intervals(pois, userVisits)

    global theme2num
    global num2theme
    global poi2theme
    theme2num, num2theme, poi2theme = get_themes_ids(pois)

    bwt_model = bwt_train_city(args.city, pois, userVisits)
    #print( bwt_model.predict(['1,12']))
    log.debug("line %d generated bwt_model: [%s]", LINE(), bwt_model)
    
    print( "--> LINE {}, bwt_ test_ city(city, pois, bwt_model, testVisits) --> {}".format(LINE(), str(bwt_model)))


    
    print(f"city: {city}, userID training : { len(userVisits['userID'].unique()) }")
    print(f"city: {city}, userID testing  : { len(testVisits['userID'].unique()) }")
    print(f"city: {city}, tryjectories for training : { len(userVisits['seqID'].unique())}")
    print(f"city: {city}, tryjectories for testing  : { len(testVisits['seqID'].unique())}")

    SUMMARY = bwt_test_city(args.city, pois, bwt_model, testVisits, times)
    SUMMARY['city']    = args.city
    for key in sorted(SUMMARY.keys()):
        print(f"\t{SUMMARY['city']}\tSUMMARY.{key}: \t{SUMMARY[key]}")
    print('  --- END OF EXECUTION / {}  ---'.format(SUMMARY['city']))
    quit(0)

def get_model_args(city, pois, train_df):
    model_args = ClassificationArgs(

    )

    model_args.reprocess_input_data = True,
    model_args.overwrite_output_dir = True
    model_args.use_multiprocessing = True

    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.001
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 5

    model_args.use_multiprocessing = True
    model_args.use_multiprocessing_for_evaluation = True
    model_args.use_multiprocessed_decoding = True

    model_args.no_deprecation_warning=True

    ### output / save disk space
    model_args.save_steps = -1
    model_args.save_model_every_epoch = False
    model_args.output_dir = "/var/tmp/1004986/output/output_{}".format(city)
    model_args.overwrite_output_dir = True

    #### PRINT WHOLE DATA TABLE
    #pd.set_option('display.max_rows', None)
    log.info("LINE {}, {} TRAINING POIs:\n{}\n\n".format(LINE(), city, str(pois)))
    log.info("LINE {}, {} TRAINING DATA:\n{}\n\n".format(LINE(), city, str(train_df)))
    log.info("LINE {}, {} TRAINING PARAMS: {}".format(LINE(), city, str(model_args)))

    model_args.early_stopping_delta = 0.0001
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 10000
    return model_args

def bwt_model(city, pois, array):

    npois = pois['poiName'].count()
    ### BERT
    NUM_LABELS= 1+ npois+ len(theme2num)
    ## [0...last_poi_id] + [theme]
    LABEL_TYPE='bert'
    MODEL_NAME='bert-base-uncased'
    USE_CUDA=False

    train_data=[]

    #print( "array : ", array)
    for arr in array: print(arr)

    for items in array:
        #print(f"items : {items}, len= {len(items)}")
        assert(len(items)>= 2)
        listA = items[:-1]
        #print("\n items: ", items)
        #print("   listA: ", listA)

        strlistA = [str(i) for i in listA]
        resultA = items[-2]
        resultB = items[-1]
        #print("  resultA  : ", resultA)
        trainItem=",".join(strlistA)
        #print("  trainItem , ", resultA)

        train_data.append( [ ",".join(strlistA) , str(resultB) ] )
        print(f" {listA} --> {resultB}")

    print(" debug: array      : ", array)
    print(" debug: train_data : ", train_data)

    for datum in train_data:
        #print(datum)
        assert(len(datum) % 2 == 0)

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["pois", "nextpoi"]
    test_df = pd.DataFrame(train_data)
    test_df.columns = ["pois", "nextpoi"]

    print("train_data[0]:", train_data[0])
    print("train_data[1]:", train_data[1])
    #print("train_df:\n",train_df)

    model = Subseq(len(train_data))
    model.fit(train_data)

    print("BWT model: ", model)
    '''
    model_args = get_model_args(city, pois, train_df)
    model = ClassificationModel(model_type= LABEL_TYPE, \
                                model_name= MODEL_NAME, \
                                num_labels= NUM_LABELS, \
                                use_cuda= USE_CUDA,\
                                #overwrite_output_dir=True, \
                                args= model_args)
    model.train_model(train_df, no_deprecation_warning=True, overwrite_output_dir=True)
    print("\nLINE {}, TRAINING MODEL: {}\n".format(LINE(), str(model)))
    print("\nLINE {}, TRAINING ARGS: {}".format(LINE(), str(model_args)))
    '''
    return model

def getTrajectories(pois, userVisits):
    trajectories=[]
    ### TRAINING DATA
    for seqid in userVisits['seqID'].unique():
        #print(seqid)

        seqtable = userVisits[ userVisits['seqID'] == seqid ]
        seqtable.sort_values(by=['dateTaken'])
        #print(seqtable['poiID'])

        pids = list(seqtable['poiID'])
        # remove duplicate
        pids = list(dict.fromkeys(pids))

        #sentense_list.append(pids)
        trajectories.append(pids)
    return trajectories

def bwt_train_city(city, pois, userVisits):

    sentense_list=[]
    list_seqs = []
    trajectories = getTrajectories(pois, userVisits)

    print(trajectories)
    # prepare training data
    for trajectory in trajectories:
        '''
        list_seqs.append(trajectory)
        '''
        n=len(trajectory)
        for head in range(0,n-1):
            for seqlen in range(2,n-head+1):
                subseq=trajectory[head:head+seqlen]
                subseq2=[]
                for pid in subseq:
                    subseq2.append(pid)
                    #subseq2.append(poi2theme[pid])
                list_seqs.append(subseq2)
                #print("--> SubSeq with Themes: ", subseq2)

    model = bwt_model(city, pois, array=list_seqs)
    log.info("MODEL: ", model)
    # predictions, raw_outputs = model.predict( to_predict=["1 2 3 4", "2 3"] )
    #print( model.predict(['1']))
    #print( model.predict(['1,12']))
    return model

def predict_mask_pos(model, seq, maskpos):
    print(f"-- predict_ mask_pos ( <model>, seq= {seq}, maskpos= {maskpos}  )")
    numpois=len(poi2theme)
    numthemes=len(theme2num)
    print(f"predict_mask_pos(model, seq='{seq}' )")
    strseq=[ str(i) for i in seq ]
    predstr=",".join(strseq)
    prediction = model.predict(predstr)
    print(f"model.predict( '{predstr}' ) ==> {prediction}")
    assert(0)
    #print("allthemes : ", allthemes)
    #assert(0)
    return (amax,amaxval,unmasked_pois)

def predict_mask(model,predseq):
    predseq_str=[str(i) for i in predseq]
    print(f"LINE {LINE()}, model.predict( to_predict=[{predseq_str}] ) ")

    strseq=[ str(i) for i in predseq ]
    predstr=",".join(strseq)
    print(f"model.predict( '{predstr}' )...")
    prediction = model.predict(predstr)
    print(f"model.predict( '{predstr}' ) ==> {prediction}")

    assert(0)
    
    possible_unmasked={}

    for maskpos in range(1,len(predseq)):
        print(f"LINE {LINE()}  predseq => {predseq},  maskpos => {maskpos}")
        nextpoi, nextval, unmasked_seq = predict_mask_pos(model,predseq, maskpos)
        print(f"LINE {LINE()}  result: nextpoi:{nextpoi} nextval:{nextval} unmasked_seq:{unmasked_seq}")
        #if maskpos and nextpoi and nextval > -999999:
        if maskpos and nextpoi and nextval > 0:
            possible_unmasked[maskpos] = nextpoi, maskpos, nextval, unmasked_seq
            print(f"  POSSIBLE: {nextval} -> {predseq} {unmasked_seq}")

    possible_unmasked= dict( sorted(possible_unmasked.items(), key=lambda item: item[1], reverse=True))
    if len(possible_unmasked) > 0:
        #print(f"  LINE {LINE()} : SORTED POSSIBLE MARKED : ", possible_unmasked)
        assert(len(possible_unmasked) > 0)
        for key in possible_unmasked:
            #print(f"  LINE {LINE()} RETURNING possible_unmasked[ {key} ] ==> {possible_unmasked[key]} ")
            nextpoi, maskpos, nextval, unmasked_seq = possible_unmasked[key]
            return nextpoi, maskpos, nextval, unmasked_seq
    log.error("LINE %d -- no prediction is found for [%s]", LINE(), predseq)
    return None,None,None,None

def estimate_duration(predseq, boot_times):
    total_duration = 0
    print(f"line {LINE()}... estimate_duration( pois ) : {predseq} ) ")
    print(f"line {LINE()}... boot_times : {boot_times} ) ")

    for p in predseq:
        if type(p) == int:
            print("boot_times : ", boot_times)
            if p in boot_times:
                intertimes = boot_times[p]
            else:
                intertimes = [1, 5 * 60] ## 5 minites

            print(" intertimes : ", intertimes)
            duration = max(intertimes)
            print(" duration   : ", duration)
            total_duration += duration
            print(f"line {LINE()}...  estimate_duration : {p} \t sec: ", int(duration))
    print(f"line {LINE()}...  total duration {predseq}  {int(total_duration)} / {int(total_duration/60)} min ")
    return total_duration

def bwt_test_city(city, pois, model, testVisits, boot_times) :
    num_pois= pois['poiID'].count()
    f1scores, recall_scores, precision_scores=[],[],[]
    micro_recall,micro_precision,micro_n = [],[],[]

    #unmasker = pipeline('fill-mask', model='bert-base-uncased')

    for seqid in testVisits['seqID'].unique():
        log.info("bwt_ test_city('%s'), seqid : %d",city,seqid)
        print(f"bwt_ test_city({city}, <pois>, <model>,...) seqid:{seqid}")

        testVisits_seq = testVisits[ testVisits['seqID']==seqid ]
        testVisits_seq.sort_values(by=['dateTaken'])
        history = testVisits_seq['poiID'].unique()
        dateTaken = testVisits_seq['dateTaken']

        seqid_duration = dateTaken.max()-dateTaken.min()+1
        p1=int(history[0])
        pz=int(history[-1])

        ### PREDICT SEQUENCE: predseq
        #predseq=[p1,pz]
        predseq=[p1]
        for iter in range(num_pois):
            print(f"\n\n-----------------\nline {LINE()} bwt_ test_ city[ {city} ] (iter: {iter}) -- predict_ mask([model], '{predseq}')")

            strseq=[ str(i) for i in predseq ]
            predstr=",".join(strseq)
            nextpoi = model.predict(predstr)

            if not nextpoi: break ### cannot predict next poi

            ## estimate duratiion of new_predseq
            predseq.append(int(nextpoi))
            print(f"line {LINE()}.. predseq => {predseq}")
        ######
        # having prediction from BWT, go to destination, pz
        ######
        if pz in predseq: predseq.append(pz)

        p,r,f = f1_scores(history, predseq)
        print(" (seqID:{}) ->  history : {}".format( seqid, history))
        print(" (seqID:{}) ->  predseq : {}".format( seqid, predseq))

        #micro_intercept = int( r * len(history) )
        micro_intercept = set(history).intersection(set(predseq))
        micro_recall.append( len(micro_intercept) / len(history) )
        micro_precision.append( len(micro_intercept) / len(predseq))
        micro_n.append(len(history))

        f1scores.append(f)
        recall_scores.append(r)
        precision_scores.append(p)
        print(" (seqID:{}) => recall:    {} %".format(seqid, r*100))
        print(" (seqID:{}) => precision: {} %".format(seqid, p*100))
        print(" (seqID:{}) => f1: {} %\t recall: {} \t precision: {}".format(seqid, f*100, r*100, p*100))

    #print("\n\nLINE {}, CALCUATUBG MICRO SCORES from {} samples".format(LINE(), sum(micro_n)))
    sum_micro_n         = np.sum(micro_n)
    sum_micro_recall    = np.sum(micro_recall)
    sum_micro_precision = np.sum(micro_precision)
    micro_precision     = sum_micro_precision / sum_micro_n
    micro_recall        = sum_micro_recall / sum_micro_n
    micro_f1            = (2*micro_precision*micro_recall) / (micro_precision+micro_recall)

    f1score_pct = 100*np.mean(f1scores)
    f1score_err = 100*np.std(f1scores)
    precision_pct = 100*np.mean(precision_scores)
    precision_err = 100*np.std(precision_scores)
    recall_pct = 100*np.mean(recall_scores)
    recall_err = 100*np.std(recall_scores)

    summary = dict()
    summary['f1score_pct'] = f1score_pct
    summary['f1score_err'] = f1score_err
    summary['precision_pct'] = precision_pct
    summary['precision_err'] = precision_err
    summary['recall_pct'] = recall_pct
    summary['recall_err'] = recall_err

    summary['micro_precision'] = 100*micro_precision
    summary['micro_recall'] = 100*micro_recall
    summary['micro_f1'] = 100*micro_f1
    return summary

def get_themes_ids(pois):
    theme2num=dict()
    num2theme=dict()
    poi2theme=dict()
    numpois = pois['poiID'].count()

    allthemes=sorted(pois['theme'].unique())
    for i in range(len(allthemes)) :
        theme2num[allthemes[i]] = i
        num2theme[i] = allthemes[i]
    print(theme2num)
    print(num2theme)
    print(allthemes)

    arr1 = pois['poiID'].array
    arr2 = pois['theme'].array

    for i in range(len(arr1)):
        pid   = arr1[i]
        theme = arr2[i]
        poi2theme[pid] = theme
        if theme not in theme2num.keys():
            num = numpois + len(theme2num.keys())
            theme2num[theme] = num
            num2theme[num] = theme

    print("\n theme2num : ", theme2num)
    print("\n num2theme : ", num2theme)
    print("\n poi2theme : ", poi2theme)
    return theme2num, num2theme, poi2theme

def main():
    for city in  ['Buda', 'Edin', 'Melb', 'Pert', 'Delh', 'Glas', 'Osak', 'Toro', 'Vien']:
        run_city(city)

if __name__ == '__main__':
    main()

