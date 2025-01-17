import argparse
import pandas as pd
import numpy as np
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

def main():
    from Bootstrap import inferPOITimes2

    parser = argparse.ArgumentParser()

    parser.add_argument('--city', '-c', type=str, required=True)
    parser.add_argument('--epoches', '-e', type=int, required=True)
    #parser.add_argument('--y', type=int, required=False)

    args = parser.parse_args()
    #print('### PYTHON:     {}'.format(args))
    print('### ARGS:       {}'.format(str(args)))
    #print('### ARGS: city: {}'.format(args.city))

    # read in from  spmf.sh /
    ### for city in ['Buda','Delh','Edin','Glas','Osak','Pert','Toro']
    (pois, userVisits, testVisits, boot_times) = load_files( args.city )

    times = calc_time_intervals(pois, userVisits)
    print(times)

    #boot_times = inferPOITimes2(pois, userVisits, alpha_pct=90)
    global theme2num
    global num2theme
    global poi2theme
    theme2num, num2theme, poi2theme = get_themes_ids(pois)

    s2sModel = bertseq2seq_train_city(args.city, pois, userVisits,testVisits, boot_times,args.epoches)

    print( "--> LINE {}, bert_test_city(city, pois, s2sModel, testVisits) --> {}".format(LINE(), str(s2sModel)))

    SUMMARY = bert_test_city(args.city, pois, s2sModel, testVisits, times)
    SUMMARY['city']    = args.city
    SUMMARY['Epoches'] = args.epoches
    for key in sorted(SUMMARY.keys()):
        print(f"\t{SUMMARY['city']}\t{SUMMARY['Epoches']} \tSUMMARY.{key}: \t{SUMMARY[key]}")

    print('  --- END OF EXECUTION / {} / {} ---'.format(SUMMARY['city'],SUMMARY['Epoches']))
    quit(0)

def get_model_args(city, epochs, pois, train_df):
    model_args = ClassificationArgs(

    )

    model_args.num_train_epochs = epochs
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
    #model_args.output_dir = "output/output_{}_e{}".format(city,epochs)
    model_args.output_dir = "/var/tmp/1004986/seq2seq_output/output_{}_e{}".format(city,epochs)
    # seq2seq_output
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

def train_city_bert_model(city, pois, array, epochs,theme2num):

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
        print("items : ", items)
        print("items : ", len(items))
        assert(len(items)>= 2)
        listA = items[:-2]
        print("\n items: ", items)
        print("   listA: ", listA)

        strlistA = [str(i) for i in listA]
        resultA = items[-2]
        resultB = items[-1]
        #print("  resultA  : ", resultA)

        trainItem=",".join(strlistA)
        print("  trainItem , ", resultA)

        train_data.append( [ trainItem, resultA ] )
    print(" debug: array            : ", array)
    print(" debug: train_data : ", train_data)
    for datum in train_data:
        print(datum)
        assert(len(datum) % 2 == 0)
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["pois", "nextpoi"]
    test_df = pd.DataFrame(train_data)
    test_df.columns = ["pois", "nextpoi"]
    
    print("train_df:\n",train_df)

    model_args = get_model_args(city,epochs, pois,train_df)

    model = ClassificationModel(model_type= LABEL_TYPE, \
                                model_name= MODEL_NAME, \
                                num_labels= NUM_LABELS, \
                                use_cuda= USE_CUDA,\
                                #overwrite_output_dir=True, \
                                args= model_args)
    model.train_model(train_df, no_deprecation_warning=True, overwrite_output_dir=True)
    print("\nLINE {}, TRAINING MODEL: {}\n".format(LINE(), str(model)))
    print("\nLINE {}, TRAINING ARGS: {}".format(LINE(), str(model_args)))
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

def bertseq2seq_train_city(city, pois, userVisits,testVisits, boot_times,epochs):

    sentense_list=[]
    list_subseqs = []
    trajectories = getTrajectories(pois, userVisits)

    print(trajectories)
    # prepare training data
    for trajectory in trajectories:

        n=len(trajectory)
        #print("trajectory : ", trajectory)
        for head in range(0,n-1):
            for seqlen in range(2,n-head+1):
                subseq=trajectory[head:head+seqlen]
                subseq2=[]
                for pid in subseq:
                    subseq2.append(pid)
                    #subseq2.append(poi2theme[pid])
                list_subseqs.append(subseq2)
                #print("--> SubSeq with Themes: ", subseq2)
    theme2num, num2theme, poi2theme = get_themes_ids(pois)
    model = train_city_bert_model(city, pois, array=list_subseqs, epochs=epochs,theme2num=theme2num)
    log.info("MODEL: ", model)
    #predictions, raw_outputs = model.predict( to_predict=["1 2 3 4", "2 3"] )
    # ADD THIS LINE to call bert_test_city() to test the model after training
    summary = bert_test_city(city, pois, model, testVisits, boot_times, poi2theme=poi2theme, theme2num=theme2num)
    return model

def predict_mask_pos(model, seq, maskpos,poi2theme,theme2num):
    print(f"-- predict_ mask_pos ( <model>, seq= {seq}, maskpos= {maskpos}  )")
    numpois=len(poi2theme)
    numthemes=len(theme2num)

    if True: ## predict next POI
        ## GIVEN argmax_theme
        ## next predict POI-ID given theme
        maskedseq=[]

        print(f"line {LINE()} ... seq: {seq}")
        ## PART-1
        for poi in seq[:maskpos]:
            # theme
            maskedseq.append( poi )
            #maskedseq.append( poi2theme[poi] )
        # PART-2
        maskedseq.append('[MASK]')
        # PART-3
        for poi in seq[maskpos:]:
            maskedseq.append( poi )
            #maskedseq.append( poi2theme[poi] )

        #print("maskedseq : ", maskedseq)
        print("      seq : ", seq)
        predict_str=",".join([ str(i) for i in maskedseq ])
        print(f"LINE {LINE()} -- model.predict( to_predict=['{predict_str}'] )")
        predictions, raw_outputs = model.predict( to_predict=[predict_str] )
        prediction=predictions[0]
        raw_output=raw_outputs[0]

    ### STEP-1  ... SKIP VISITED POIs
    for i in range(len(seq)):
        poi=seq[i]
        raw_output[poi] = -999999

    ### STEP-2  ... LOOK UP only POIS in raw_ouput
    poi_raw_output = raw_output[1:numpois]

    ### STEP-3  ... DECIDE argmax in POI output
    ## amax starts from poi-0
    amax = 1+int(np.argmax(poi_raw_output))

    amaxval = raw_output[amax]
    print(f"  seq     -> {seq}")
    print(f"  amax    -> {amax}")
    print(f"  amaxval -> {amaxval}")
    #print(f"maskedseq => {maskedseq}")
    print(f"maskpos   => {maskpos}")

    if amaxval <= -999999:
        ### when predicted (amax) is already in seq
        ### there is no more POIs to predict,
        return None, None, None

    assert(amax not in seq)
    assert(maskedseq[maskpos] == '[MASK]')
    unmasked = maskedseq.copy()
    unmasked[maskpos] = amax

    #print(f"--> predict_ mask_pos (model, '{seq}', {maskpos} ) --- amax: {amax}")
    #print(f"--> predict_ mask_pos (model, '{seq}', {maskpos} ) --- masked:\n {maskedseq}")
    #print(f"--> predict_ mask_pos (model, '{seq}', {maskpos} ) --- poi2theme \n {poi2theme} ")

    assert(amax in poi2theme.keys())
    amax_theme = poi2theme[amax]
    #unmasked.insert(maskpos*2+1, amax_theme)
    print(f"--> predict_ mask_pos (model, '{seq}', {maskpos} ) --- {unmasked} ")

    #print("theme2num : ", theme2num)
    #print("num2theme : ", num2theme)

    unmasked_pois=unmasked[0::2]
    #print("allthemes : ", allthemes)
    #assert(0)
    return (amax,amaxval,unmasked_pois)

def predict_mask(model,predseq,poi2theme,theme2num):
    predseq_str=[str(i) for i in predseq]
    print(f"LINE {LINE()}, model.predict( to_predict=[{predseq_str}] ) ")

    possible_unmasked={}

    for maskpos in [len(predseq)-1]:
    #for maskpos in range(1,len(predseq)):
        print(f"LINE {LINE()}  predseq => {predseq},  maskpos => {maskpos}")
        nextpoi, nextval, unmasked_seq = predict_mask_pos(model,predseq, maskpos,poi2theme,theme2num)
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
            duration = int(np.mean(intertimes))
            print(" duration   : ", duration)
            total_duration += duration
            print(f"line {LINE()}...  estimate_duration : {p} \t sec: ", int(duration))
    print(f"line {LINE()}...  total duration {predseq}  {int(total_duration)} / {int(total_duration/60)} min ")
    return total_duration

def bert_test_city(city, pois, model, testVisits, boot_times,poi2theme,theme2num) :
    num_pois= pois['poiID'].count()
    f1scores, recall_scores, precision_scores=[],[],[]
    micro_recall,micro_precision,micro_n = [],[],[]

    #unmasker = pipeline('fill-mask', model='bert-base-uncased')

    for seqid in testVisits['seqID'].unique():
        #log.info("bert_test_city('%s'), seqid : %d",city,seqid)
        testVisits_seq = testVisits[ testVisits['seqID']==seqid ]
        testVisits_seq.sort_values(by=['dateTaken'])
        history = testVisits_seq['poiID'].unique()
        dateTaken = testVisits_seq['dateTaken']

        seqid_duration = dateTaken.max()-dateTaken.min()+1
        p1=int(history[0])
        pz=int(history[-1])

        ### PREDICT SEQUENCE: predseq
        predseq=[p1,pz]
        for iter in range(num_pois):
            ## print("predict : ", predictions)
            ## print("raw_outputs >>\n", raw_outputs)
            ### INPUT: predseq
            print(f"\n\n-----------------\nline {LINE()} bert_test_city[ {city} ] (iter: {iter}) -- predict_ mask([model], '{predseq}')")
            nextpoi, maskpos, nextval, unmasked_seq = predict_mask(model, predseq,poi2theme,theme2num)
            print(f"line {LINE()} bert_test_city[ {city} ] (iter: {iter}) -- predict_ mask([model], '{predseq}')")
            print(f"line {LINE()} nextpoi:{nextpoi}, maskpos:{maskpos}, nextval:{nextval}, unmasked_seq:{unmasked_seq}")

            if not nextpoi: break ### cannot predict next poi

            ## estimate duratiion of new_predseq
            predseq = unmasked_seq
            poi_duration = estimate_duration(unmasked_seq, boot_times  )
            print(f"LINE:{LINE()} (iter: {iter}) -- predict_mask([model], '{predseq}' ->  '{poi_duration} < {seqid_duration} sec' ?)")
            if poi_duration > seqid_duration: break

        #print(f"  seqid:{seqid} HISTORY:        ",history)
        #print(f"  seqid:{seqid} HISTORY TIME:   ",seqid_duration)
        #print(f"  seqid:{seqid} predictION:     ",predseq)
        #print(f"  seqid:{seqid} HISTORY TIME:   ",poi_duration)

        ### same length for predseq and history
        #predseq=predseq[0:len(history)]
        strarr_history=[ str(i) for i in history ]
        strarr_predseq=[ str(i) for i in predseq ]
        #print("LINE {} => history: {}".format( LINE(), str(" -> ".join(strarr_history))))
        #print("LINE {} => predseq: {}".format( LINE(), str(" -> ".join(strarr_predseq))))


        ### f1 score
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
    sum_micro_n = sum(micro_n)
    sum_micro_recall = sum(micro_recall)
    sum_micro_precision = sum(micro_precision)

    micro_precision = sum_micro_precision / sum_micro_n
    micro_recall    = sum_micro_recall / sum_micro_n
    micro_f1        = (2*micro_precision*micro_recall) / (micro_precision+micro_recall)

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

    #summary['micro_precision'] = 100*micro_precision
    #summary['micro_recall'] = 100*micro_recall
    #summary['micro_f1'] = 100*micro_f1
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

if __name__ == '__main__':
    main()