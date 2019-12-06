/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataminingproject;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Lorenzo
 */
public class ClassifierHandler {

    public void trainAllClassifiers(Classifier cl, String classifierName) {
        try {
            for (int i = 0; i < Parameters.K_FOLD; i++) {
                ConverterUtils.DataSource srcTest = new ConverterUtils.DataSource("traintest_sets\\test" + i + ".arff");
                ConverterUtils.DataSource srcTrain = new ConverterUtils.DataSource("traintest_sets\\train" + i + ".arff");

                Instances test = srcTest.getDataSet();
                Instances train = srcTrain.getDataSet();

                cl = trainClassifier(cl, train);
                Evaluation ev = evaluateClassifier(cl, train, test);

                SerializationHelper.write("models\\"+classifierName+"\\model" + i, cl);
                SerializationHelper.write("models\\"+classifierName+"\\evaluation" + i, ev);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    

    public Classifier trainClassifier(Classifier cl, Instances train) {
        try {
            train.setClassIndex(25);
            
            System.out.println(train.classAttribute());

            String[] opt = new String[]{"-R", "2,8"};
            Remove remove = new Remove();
            remove.setOptions(opt);
            remove.setInputFormat(train);
            train = Filter.useFilter(train, remove);
            

            cl.buildClassifier(train);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return cl;
    }

    public Evaluation evaluateClassifier(Classifier cl, Instances train, Instances test) {
        Evaluation eval = null;

        try {
            train.setClassIndex(25);
            test.setClassIndex(25);

            String[] opt = new String[]{"-R", "2,8"};
            Remove remove = new Remove();
            remove.setOptions(opt);
            remove.setInputFormat(train);
            train = Filter.useFilter(train, remove);

            remove.setInputFormat(test);
            test = Filter.useFilter(test, remove);

            eval = new Evaluation(train);
            eval.evaluateModel(cl, test);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return eval;
    }

    public void combinedEvaluation(String classifierName) {
        try {
            List<Evaluation> evals = new ArrayList<Evaluation>();
            double corrCoeff = 0;
            double mae = 0;
            double rmse = 0;
            double rae = 0;
            double rrse = 0;
            double tni = 0;
            for (int i = 0; i < Parameters.K_FOLD; i++) {
                evals.add((Evaluation) SerializationHelper.read("models\\"+classifierName+"\\evaluation" + i));

                corrCoeff += evals.get(i).correlationCoefficient();
                mae += evals.get(i).meanAbsoluteError();
                rmse += evals.get(i).rootMeanPriorSquaredError();
                rae += evals.get(i).relativeAbsoluteError();
                rrse += evals.get(i).rootRelativeSquaredError();
                tni += evals.get(i).numInstances();

                System.out.println(evals.get(i).toSummaryString());
                System.out.println("\n\n\n");
            }
            System.out.println("Average Correlation Coefficient: " + corrCoeff / Parameters.K_FOLD + "\n"
                    + "Average Mean Absolute Error: " + mae / Parameters.K_FOLD + "\n"
                    + "Average Root Mean Squared Error: " + rmse / Parameters.K_FOLD + "\n"
                    + "Average Relative Absolute Error: " + rae / Parameters.K_FOLD + "\n"
                    + "Average Root Relative Squred Error: " + rrse / Parameters.K_FOLD + "\n"
                    + "Average instances: " + tni / Parameters.K_FOLD + "\n");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
}
