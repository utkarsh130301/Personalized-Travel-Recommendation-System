import argparse
import pandas as pd
import numpy as np
from common import f1_scores, LINE, log
from poidata import (
    load_files,getThemes,getPOIFullNames,getPOIThemes,poi_name_dict)

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs,
    MultiLabelClassificationModel, MultiLabelClassificationArgs)

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
    model_args.output_dir = "/var/tmp/1004986/output/output_{}_e{}".format(city,epochs)
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

def train_city_bert_model(city, pois, array, epochs):
    npois = pois['poiName'].count()
    ### BERT
    NUM_LABELS=npois+1 ## [0...last_poi_id]
    LABEL_TYPE='bert'
    MODEL_NAME='bert-base-uncased'
    USE_CUDA=False

    train_data=[]
    for items in array:
        #print("  items: ", items)
        listA = items[:-1]
        strlistA = [str(i) for i in listA]
        resultA = items[-1]

        trainItem=",".join(strlistA)
        train_data.append( [trainItem, int(resultA)] )

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["pois", "nextpoi"]
    test_df = pd.DataFrame(train_data)
    test_df.columns = ["pois", "nextpoi"]


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

def getTrajectories(userVisits):
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

def bert_train_city(city, pois, userVisits, epochs):
    sentense_list=[]
    list_subseqs = []
    trajectories = getTrajectories(userVisits)

    # prepare training data
    for trajectory in trajectories:

        n=len(trajectory)
        #print("trajectory : ", trajectory)
        for head in range(0,n-1):
            for seqlen in range(2,n-head+1):
                subseq=trajectory[head:head+seqlen]
                #print("  [ sebseq: {} head:{} seqlen:{}".format(subseq,head,seqlen))
                list_subseqs.append(subseq)

    model = train_city_bert_model(city, pois, array=list_subseqs, epochs=epochs)
    #print("MODEL: ", model)
    ## pred_ictions, raw_outputs = model.pred_ict( to_pred_ict=["1 2 3", "2 3"] )
    ## print("pred_ict : ", pred_ictions)
    ## print("raw_outputs >>\n", raw_outputs)
    return model

def predict_mask_pos(model, seq, maskpos):
    print(f"LINE {LINE()} predict_mask_pos(<model>, {seq}, {maskpos})  ")
    assert(maskpos < len(seq))

    maskedseq = seq[:maskpos]
    maskedseq.append('[MASK]')
    maskedseq.extend( seq[maskpos:] )
    print(f"LINE_133 maskedseq: {maskedseq}")

    assert(len(maskedseq)==len(seq)+1)

    print(f"LINE_133 maskedseq: {maskedseq}")

    predict_str=",".join([str(i)  for i in maskedseq])
    #print(f"LINE_133 predict_str: {predict_str}")

    predictions, raw_outputs = model.predict( to_predict=[predict_str] )
    prediction=predictions[0]
    raw_output=raw_outputs[0]

    print(f"line {LINE()}  predict_str : {predict_str}")
    print(f"line {LINE()}  raw_output : {raw_output}")
    ## ignore all pois aready in seq

    for i in range(len(seq)):
        p=seq[i]
        raw_output[p] = -999999

    print(f"line {LINE()} -99999 raw_output : \n{raw_output}")

    amax = int(np.argmax(raw_output))
    amaxval = raw_output[amax]
    print(f"  amax -> {amax}")
    print(f"  seq  -> {seq}")
    assert(amax not in seq)
    assert(maskedseq[maskpos] == '[MASK]')

    unmasked = maskedseq.copy()
    assert(unmasked[maskpos] == '[MASK]')
    unmasked[maskpos]=amax
    return (amax,amaxval,unmasked)

def predict_mask(model,predseq):
    predseq_str=[str(i) for i in predseq]
    #predict_str=",".join(predseq_str)
    #log.info("LINE {}, iteration:{} , to_predict: [{}] ".format(LINE(),i, predict_str))
    print(f"LINE {LINE()}, model.predict( to_predict=[{predseq_str}] ) ")

    possible_unmasked={}
    for maskpos in range(1,len(predseq)):

        print(f"LINE {LINE()}  predseq => {predseq},  maskpos => {maskpos}")
        nextpoi, nextval, unmasked_seq = predict_mask_pos(model,predseq, maskpos)
        print(f"RESULT: {nextpoi} {nextval} {unmasked_seq}")

        if nextval > -999999:
            possible_unmasked[maskpos] = nextpoi, maskpos, nextval, unmasked_seq
            print(f"POSSIBLE: {nextval} -> {predseq} {unmasked_seq}")

    possible_unmasked= dict( sorted(possible_unmasked.items(), key=lambda item: item[1], reverse=True))


    print(f"LINE {LINE()} : SORTED POSSIBLE MARKED : ", possible_unmasked)
    assert(len(possible_unmasked) > 0)    

    for key in possible_unmasked:
        print(f"LINE {LINE()}  possible_unmasked[ {key} ] ==> {possible_unmasked[key]} ")
        # retur: (0.46243739128112793, ['17', '12', '9'])
        return possible_unmasked[key]
    assert(0)

def estimate_duration(predseq, boot_times):
    total_duration = 0
    print(f"line {LINE()}... estimate_duration(  {predseq} ) ")

    for p in predseq:

        if p in boot_times:
            intertimes = boot_times[p]
        else:
            intertimes = [1, 5 * 60] ## 5 minites

        duration = max(intertimes)
        total_duration += duration
        print(f"line {LINE()}...  p: {p} \t sec: ", int(duration))
    print(f"line {LINE()}...  total duration {predseq}  {int(total_duration)} / {int(total_duration/60)} min ")
    return total_duration

def bert_test_city(city, pois, model, testVisits, boot_times) :
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
        #dateTakenInt= [ (int) for d in dateTaken]
        #timestamp1= testVisits_seq['dateTaken'].min(dateTakenInt)
        #timestamp2= testVisits_seq['dateTaken'].max(dateTakenInt)
        seqid_duration = dateTaken.max()-dateTaken.min()+1
        p1=int(history[0])
        pz=int(history[-1])

        ### PREDICT SEQUENCE: predseq
        predseq=[p1,pz]
        for iter in range(num_pois):
            ## print("predict : ", predictions)
            ## print("raw_outputs >>\n", raw_outputs)
            ### INPUT: predseq
            print("(iter: {}) -- predict_mask([model], '{}')".format(iter,predseq))
            nextpoi, maskpos, nextval, unmasked_seq = predict_mask(model, predseq)

            print("LINE {} predict_mask(model, predseq) -> nextpoi:{}, maskpos:{}, nextval:{}, unmasked_seq:{}"\
                .format(LINE(), nextpoi, maskpos, nextval, unmasked_seq))

            ## estimate duratiion of new_predseq
            predseq = unmasked_seq
            poi_duration = estimate_duration(unmasked_seq, boot_times)
            print("(iter: {}) -- predict_mask([model], '{}' -> {} sec)".format(iter,predseq,poi_duration))

            if poi_duration > seqid_duration: break

        print(f"  seqid:{seqid} HISTORY:        ",history)
        print(f"  seqid:{seqid} HISTORY TIME:   ",seqid_duration)
        print(f"  seqid:{seqid} PREDITION:      ",predseq)
        print(f"  seqid:{seqid} HISTORY TIME:   ",poi_duration)

        ### same length for predseq and history
        predseq=predseq[0:len(history)]
        strarr_history=[ str(i) for i in history ]
        strarr_predseq=[ str(i) for i in predseq ]
        print("LINE {} => history: {}".format( LINE(), str("->".join(strarr_history))))
        print("LINE {} => predseq: {}".format( LINE(), str("->".join(strarr_predseq))))


        ### F1 score
        p,r,f = f1_scores(history, predseq)

        print(" (eval_seqid: {}  history => {}".format( seqid, history))
        print(" (eval_seqid: {}  predseq => {}".format( seqid, predseq))

        micro_intercept = int( r * len(history) )
        micro_recall.append( int(r * len(history) ))
        micro_precision.append( int(p * len(predseq)) )
        micro_n.append(len(history))

        f1scores.append(f)
        recall_scores.append(r)
        precision_scores.append(p)
        print(" (seqID:{}) => recall:    {} %".format(seqid, r*100))
        print(" (seqID:{}) => precision: {} %".format(seqid, p*100))
        print(" (seqID:{}) => F1:        {} %".format(seqid, f*100))
        print(" (seqID:{}) => F1: {} %\t recall: {} \t precision: {}".format(seqid, f*100, r*100, p*100))

    #print("\n\nLINE {}, CALCUATUBG MICRO SCORES from {} samples".format(LINE(), sum(micro_n)))
    sum_micro_n = sum(micro_n)
    sum_micro_recall = sum(micro_recall)
    sum_micro_precision = sum(micro_precision)

    micro_precision = sum_micro_precision / sum_micro_n
    micro_recall    = sum_micro_recall / sum_micro_n
    micro_f1        = (2*micro_precision*micro_recall) / (micro_precision+micro_recall)

    #print( "=> micro_precision = {}/{} = {} %".format( sum_micro_precision, sum_micro_n, 100*micro_precision ) )
    #print( "=> micro_recall    = {}/{} = {} %".format( sum_micro_recall,    sum_micro_n, 100*micro_recall ) )
    #print( "=> micro_f1        = 2*{1:3g}*{1:3g} / ( {1:3g}+{1:3g} ) = {1:3g} %" \
    #  .format(micro_precision, micro_recall,\
    #          micro_precision, micro_recall,\
    #          micro_f1))

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

def main():
    from Bootstrap import inferPOITimes

    parser = argparse.ArgumentParser()

    parser.add_argument('--city', '-c', type=str, required=True)
    parser.add_argument('--epoches', '-e', type=int, required=True)
    #parser.add_argument('--y', type=int, required=False)

    args = parser.parse_args()
    print('### PYTHON:     {}'.format("Bert_POI_predict.py"))
    print('### ARGS:       {}'.format(str(args)))
    print('### ARGS: city: {}'.format(args.city))

    # read in from  spmf.sh /
    ### for city in ['Buda','Delh','Edin','Glas','Osak','Pert','Toro']
    (pois, userVisits, testVisits, costProfCat) = load_files( args.city )
    #print( pois )

    boot_times = inferPOITimes(pois, userVisits, alpha_pct=90)

    #print( "--> LINE %d, bert_train_city(pois, userVisits) ".format(LINE()))
    bertmodel = bert_train_city(args.city, pois, userVisits, args.epoches)

    print( "--> LINE %d, bert_test_city(city, pois, model, testVisits) --> %s".format(LINE(), str(bertmodel)))
    SUMMARY = bert_test_city(args.city, pois, bertmodel, testVisits, boot_times)

    SUMMARY['city']    = args.city
    SUMMARY['Epoches'] = args.epoches
    for key in sorted(SUMMARY.keys()):
        print(f"{SUMMARY['city']} {SUMMARY['Epoches']} SUMMARY.{key}:\t{SUMMARY[key]}")

    print('  --- END OF EXECUTION / {} / {} ---'.format(SUMMARY['city'],SUMMARY['Epoches']))
    quit(0)

if __name__ == '__main__':
    main()



