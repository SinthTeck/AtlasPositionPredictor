/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataminingproject;

import java.util.ArrayList;
import java.util.List;
import sun.nio.cs.ext.IBM037;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.misc.InputMappedClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Lorenzo
 */
public class DataMiningProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        //CASE 1: Reduced dataset
        RawCSVHandler h = new RawCSVHandler(Parameters.REDUCED_DATASET_PATH);
        h.splitTrainTestSet();
        h.preprocess();
        
        ClassifierHandler ch = new ClassifierHandler();
        IBk knn = new IBk(50);
        AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
        asc.setClassifier(knn);
        ch.trainAllClassifiers(asc, "asc"); //There needs to be a folder in models called [model name]
        ch.combinedEvaluation("asc");
        
        //Testing classifiers for the 100 closest sources
        try {
            for (int i = 0; i < Parameters.K_FOLD; i++) {
                AttributeSelectedClassifier cl = (AttributeSelectedClassifier) SerializationHelper.read("models\\asc\\model"+i);
                ApplicationEvaluator e = new ApplicationEvaluator();
                e.evaluateApplication(cl, i);
                e.eval(i);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        
        //CASE 2: Complete dataset
        /*
        RawCSVHandler h = new RawCSVHandler(Parameters.COMPLETE_DATASET_PATH);
        h.splitTrainTestSet();
        h.preprocessComplete();        
        
        
        ClassifierHandler ch = new ClassifierHandler();
        AttributeSelectedClassifier asc = null;
        
        try {
            asc = new AttributeSelectedClassifier();
            asc.setClassifier(new IBk(50));
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        ch.trainAllClassifiers(asc, "ascComplete");
        ch.combinedEvaluation("ascComplete");
        
        //Testing classifiers for the 100 closest sources
        try {
            for (int i = 0; i < Parameters.K_FOLD; i++) {
                AttributeSelectedClassifier cl = (AttributeSelectedClassifier) SerializationHelper.read("models\\ascComplete\\model"+i);
                ApplicationEvaluator e = new ApplicationEvaluator();
                e.evaluateApplicationToCompleteDataset(cl, i);
                e.eval(i);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        */


    }
                

}
