//package ca.pfv.spmf.test;
 
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.util.Map;
import java.util.Map.Entry;

import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Item;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.Sequence;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.SequenceDatabase;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.database.SequenceStatsGenerator;

import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.Predictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.CPT.CPT.CPTPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.CPT.CPTPlus.CPTPlusPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.DG.DGPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.TDAG.TDAGPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.LZ78.LZ78Predictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.Markov.MarkovAllKPredictor;
import ca.pfv.spmf.algorithms.sequenceprediction.ipredict.predictor.Markov.MarkovFirstOrderPredictor;

/**
 * Example of how to use the CPT+ sequence prediction model in the source code.
 * Copyright 2015.
 */
public class SPMFPredictor {

	public SPMFPredictor(String algo, String city, String inputPath, String [] args) throws IOException, ClassNotFoundException {
		//System.err.println("\n\nLINE_27 -- SPMFPredictor(algo='"+algo+"', city='" + city + "', inputPath='" + inputPath +"' );");		
		this.args=args;
		predictionModel = getModel(algo,inputPath);
		//System.err.println("LINE_32 -- Predictor model = " + predictionModel);

		if (predictionModel == null) System.exit(-3);

	}

    private Predictor predictionModel=null;
    private String [] args=null;

    public Predictor getModel(String algo, String inputPath) throws IOException {	
		//System.err.println("LINE_38 --   getModel( " + algo + ", " + inputPath +" );");
		SequenceDatabase trainingSet = new SequenceDatabase();
		trainingSet.loadFileSPMFFormat(inputPath, Integer.MAX_VALUE, 0, Integer.MAX_VALUE);
		//System.err.println("LINE_46 --   trainingSet : " + trainingSet +" );");

		for(Sequence sequence : trainingSet.getSequences()) {
			//System.out.println("input: train_db: " + sequence.toString());
		}

		//System.out.println();
		// Print statistics about the training sequences
		//SequenceStatsGenerator.prinStats(trainingSet, " training sequences ");
					
		// The following line is to set optional parameters for the prediction model. 
		// We can activate the recursive divider strategy to obtain more noise
		// tolerant predictions (see paper). We can also use a splitting method
		// to reduce the model size (see explanation below).
		String optionalParameters = "splitLength:6 splitMethod:0 recursiveDividerMin:1 recursiveDividerMax:5";
					
		// An explanation about "splitMethod":
		// - If we set splitMethod to 0, then each sequence will be completely used
		//   for training. 
		// - If we set splitMethod to 1, then only the last k (here k = 6) symbols of
		// each sequence will be used for training. This will result in a smaller model
		// and faster prediction, but may decrease accuracy.
		// - If we set splitMethod to 2, then each sequence will be divided in several
		//   subsequences of length k or less to be used for training. 
					
		// Train the prediction model
		//System.err.println(".. line 73 ALGO  '" + algo + "'");

		if (algo.equals("CPT")) {
			System.err.println(".. line 76 ALGO  " + algo );
			predictionModel = new CPTPredictor("CPT", optionalParameters);
		} else if (algo.equals("CPTPlus")) {
			optionalParameters = "CCF:true CBS:true CCFmin:1 CCFmax:6 CCFsup:2 splitMethod:0 splitLength:4 minPredictionRatio:1.0 noiseRatio:1.0";			
			predictionModel = new CPTPlusPredictor("CPT+", optionalParameters);
			predictionModel.Train(trainingSet.getSequences());
		} else if (algo.equals("DG")) {
			optionalParameters = "lookahead:2";
			predictionModel = new DGPredictor("DG", optionalParameters);
		} else if (algo.equals("TDAG")) {
			predictionModel = new TDAGPredictor("TDAG");
		} else if (algo.equals("LZ78")) {
			predictionModel = new LZ78Predictor("LZ78");
		} else if (algo.equals("MarkovAllK")) {
		    //                        MarkovAllKPredictor
			predictionModel = new MarkovAllKPredictor("MarkovAllKPredictor", optionalParameters);
		} else if (algo.equals("MarkovFirstOrder")) {
			predictionModel = new MarkovFirstOrderPredictor("MarkovFirstOrderPredictor", optionalParameters);
		} else {
			System.err.println("UNKNOWN predictionModel: " + algo);
			System.exit(-1);
		}
		// TRAINING
		predictionModel.Train(trainingSet.getSequences());
		//System.err.println("LINE_90 -- predictionModel = " + predictionModel);
		return  predictionModel;
    }	

    public Sequence predict(Sequence headPred) {	 
		System.err.println("--   predict( " + headPred +" );");
		return null;
    }

    public Sequence getPrediction() { //Predictor predictionModel, String args[]) {	 
		//System.err.println("LINE 91 --   get-Prediction( " + predictionModel +", args );");
		if (predictionModel == null) {
			System.err.println("\nERROR\nERROR\nERROR: predictionModel : " + predictionModel+ "\nERROR\nERROR\n");
			System.exit(-1);
		}
		//System.out.println("LINE 69 --- get-Prediction(...)");
		// PRE-DICTION
			
		// Now we will use the prediction model that we have trained to make a prediction.
		// We want to predict what would occur after the sequence <1, 4>.
		// We first create the sequence
		Sequence sequence = new Sequence(0);



		//sequence.addItem(new Item(1));
		//sequence.addItem(new Item(4));
		//System.err.println("###  args.length => " + args.length );
		if (args.length <= 1) {
			System.err.println("###  java " + args[0] + " <training data> [poi_1] [poi_2] [poi_3]...");
			System.exit(-1);
		}


		java.util.Set<Integer> poiSet = new java.util.HashSet<Integer>(0); 
		String predStr="<";

		if (args.length==0) {
			System.exit(-1);
		} else {
			for (int i=3; i<args.length; i++) {
				//System.err.println("### init.poi => args[" + i + "] => " + args[i] );
				sequence.addItem(new Item(Integer.parseInt(args[i])));
				poiSet.add(Integer.valueOf(Integer.parseInt(args[i])));
			
				predStr=predStr + "("+ args[i] + ")";
			}
			predStr=predStr+">";			
		}

		System.out.println("LINE 127, predStr : " + predStr);
		//System.out.println("LINE 99 --- predStr ==> " + predStr);
		//Sequence predict(Sequence headPred) {
			
		// Then we perform the prediction
		for (int iter=0; iter<100; iter++) {
			//System.out.println("\nline 104 ### sequence: ["+ predStr +"] next symbol is: [" + sequence + "]");
			Sequence sequence2 = predictionModel.Predict(sequence);
			//System.out.println("line 127 ### sequence2: " + sequence2);

			String seq2 = sequence2.toString().replace("(", "");
			//System.out.println("line 129 ### seq2: " + seq2);
			seq2 = seq2.replace(")", "");
			//System.out.println("line 131 ### seq2: " + seq2);
			seq2 = seq2.replace(" ", "");
			//System.out.println("line 133: seq2 : " + seq2);

			if (!seq2.toString().isEmpty() && seq2.toString() != "" ) {
				Integer nextPOI = Integer.parseInt(seq2);
				if (poiSet.contains(nextPOI)) {
					//System.out.println("REPEATED... Iter:"+iter+" poiSet : " + poiSet);
					break;
				} else {
	
					poiSet.add(nextPOI);
	
					//System.out.println("    add to poiSet : " + nextPOI);
					sequence.addItem(new Item(Integer.parseInt(seq2)));
				}
				//System.out.println("LINE 113 ### sequence: " + sequence );	
			}
		}
		System.out.println("\n### Predicted sequence : ");
		return (sequence);
    }

    public static void main(String [] args) throws IOException, ClassNotFoundException {		

		// Load the set of training sequences

		String inputPath = fileToPath("contextCPT.txt");  

		String algo="";
		String city;

		if (args.length <= 1) {
			//System.err.println("SYNTAX: java " + this.getClass().getSimpleName() + "  " + args[0] + " [CPT / LZ78 / ]  [city]");
			System.err.println("SYNTAX: java SPMFPredictor  <algo>       <city> <context/txt>    <poiid>");
			System.err.println("examle: java SPMFPredictor  CPT(or LZ78) Pert   contextCPT.txt   2");
			 
			// MarkovFirstOrderPredictor / MarkovAllKPredictor / TDAGPredictor 
			// CPTPlusPredictor / CPTPredictor_POC / CPTPredictor /
			// LZ78Predictor / ALZ / DGPredictor
			System.exit(-1);
		} else {
			//System.err.println("... Using input file: " + args[0]);
			algo = args[0];
			city = args[1];
			inputPath = args[2];

			//System.err.println(".. algo -> " + algo + " .. seq: args : " + args);
			//for (int i=0; i<args.length; i++)  System.err.println(".. seq: args[ " + i + " ] -> " + args[i]);
			//System.err.println(".. SPMFPredictor spmfpreditor -> " + algo);

			SPMFPredictor spmfpreditor = new SPMFPredictor(algo, city, inputPath, args);

			//System.err.println(".. SPMFPredictor algo -> "+ algo + "spmfpreditor -> " + spmfpreditor);

		    Sequence predicted = spmfpreditor.getPrediction();
			String predStr = predicted.getItems().toString().replace("[","").replace("]","").replace(" ","");
			System.out.println(predStr);
	
			// System.out.println(sequence.toString().replace("(").replace(")"));
			// (String algo, String city, String inputPath, String [] args

		}
	}
		
	public static String fileToPath(String filename) throws UnsupportedEncodingException{
		URL url = SPMFPredictor.class.getResource(filename);
		return java.net.URLDecoder.decode(url.getPath(),"UTF-8");
    }
}

