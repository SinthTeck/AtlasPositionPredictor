package dataminingproject;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Lorenzo
 */
public class ApplicationEvaluator {

    public void evaluateApplication(Classifier cl, int index) {
        Map<String, Map<String, Double>> actualValues = new HashMap<>();
        Map<String, Map<String, Double>> predictedValues = new HashMap<>();
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("traintest_sets\\test" + index + ".arff");
            Instances test = source.getDataSet();
            test.sort(7);

            for (Instance inst : test) {
                String S_ID = inst.toString(1);
                String D_ID = inst.toString(7);
                Double RTT = Double.parseDouble(inst.toString(25));
                if (actualValues.containsKey(D_ID)) {
                    actualValues.get(D_ID).put(S_ID, RTT);
                } else {
                    actualValues.put(D_ID, new HashMap<>());
                    actualValues.get(D_ID).put(S_ID, RTT);
                }
            }

            Instances test2 = new Instances(test);
            test2.setClassIndex(25);
            test2.deleteAttributeAt(1);
            test2.deleteAttributeAt(6);

            for (int i = 0; i < test.size(); i++) {

                Instance inst = test.get(i);
                String S_ID = inst.toString(1);
                String D_ID = inst.toString(7);

                Double RTT = cl.classifyInstance(test2.get(i));

                if (predictedValues.containsKey(D_ID)) {
                    predictedValues.get(D_ID).put(S_ID, RTT);
                } else {
                    predictedValues.put(D_ID, new HashMap<>());
                    predictedValues.get(D_ID).put(S_ID, RTT);
                }
            }

            FileOutputStream fileOut
                    = new FileOutputStream("actualValues" + index + ".ser");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(actualValues);
            out.close();
            fileOut.close();

            fileOut
                    = new FileOutputStream("predictedValues" + index + ".ser");
            out = new ObjectOutputStream(fileOut);
            out.writeObject(predictedValues);
            out.close();
            fileOut.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void eval(int index) {
        try {
            FileInputStream fileIn = new FileInputStream("actualValues" + index + ".ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            Map<String, Map<String, Double>> actualValues = (Map<String, Map<String, Double>>) in.readObject();
            in.close();
            fileIn.close();

            fileIn = new FileInputStream("predictedValues" + index + ".ser");
            in = new ObjectInputStream(fileIn);
            Map<String, Map<String, Double>> predictedValues = (Map<String, Map<String, Double>>) in.readObject();
            in.close();
            fileIn.close();

            int counter = 0;
            int totCount = 0;
            int numDiscarded = 0;

            for (String key : actualValues.keySet()) {
                Map<String, Double> actual = actualValues.get(key);
                Map<String, Double> predicted = predictedValues.get(key);

                actual = sortByValue(actual);
                predicted = sortByValue(predicted);

                if (actual.size() < 100) {
                    numDiscarded++;
                    continue;
                }

                List<String> actualClosest = new ArrayList<>(actual.keySet());
                actualClosest = actualClosest.subList(0, 100);

                List<String> predictedClosest = new ArrayList<>(predicted.keySet());
                //Collections.shuffle(predictedClosest); //Uncomment to compare to random pick of sources
                predictedClosest = predictedClosest.subList(0, 100);

                for (int i = 0; i < 100; i++) {
                    if (predictedClosest.contains(actualClosest.get(i))) {
                        counter++;
                    }
                }
                System.out.println("Target " + key + ": " + counter + "/100, Total sources: " + predicted.size());
                totCount += counter;
                counter = 0;
            }
            totCount = totCount / predictedValues.keySet().size();

            System.out.println("\tClassifier n°" + index + ": " + totCount + "/100 avg, N° of discarded targets: " + numDiscarded);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void evaluateApplicationToCompleteDataset(Classifier cl, int index) {
        Map<String, Map<String, Double>> actualValues = new HashMap<>();
        Map<String, Map<String, Double>> predictedValues = new HashMap<>();
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:\\Users\\Lorenzo\\Desktop\\Data mining\\Project\\total-europe-ML-complete.arff");
            Instances test = source.getDataSet();
            test.sort(7);

            for (Instance inst : test) {
                String S_ID = inst.toString(1);
                String D_ID = inst.toString(7);
                Double GD = Double.parseDouble(inst.toString(10));
                if (actualValues.containsKey(D_ID)) {
                    actualValues.get(D_ID).put(S_ID, GD);
                } else {
                    actualValues.put(D_ID, new HashMap<>());
                    actualValues.get(D_ID).put(S_ID, GD);
                }
            }

            FileOutputStream fileOut
                    = new FileOutputStream("actualValues" + index + ".ser");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(actualValues);
            out.close();
            fileOut.close();

            Instances test2 = new Instances(test);
            test2.setClassIndex(25);
            test2.deleteAttributeAt(10);
            test2.deleteAttributeAt(1);
            test2.deleteAttributeAt(6);

            for (int i = 0; i < test.size(); i++) {

                Instance inst = test.get(i);
                String S_ID = inst.toString(1);
                String D_ID = inst.toString(7);

                Double RTT = cl.classifyInstance(test2.get(i));

                if (predictedValues.containsKey(D_ID)) {
                    predictedValues.get(D_ID).put(S_ID, RTT);
                } else {
                    predictedValues.put(D_ID, new HashMap<>());
                    predictedValues.get(D_ID).put(S_ID, RTT);
                }
            }

            fileOut
                    = new FileOutputStream("predictedValues" + index + ".ser");
            out = new ObjectOutputStream(fileOut);
            out.writeObject(predictedValues);
            out.close();
            fileOut.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void evalToCompleteDataset(int index) {
        try {
            FileInputStream fileIn = new FileInputStream("actualValues.ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            Map<String, Map<String, Double>> actualValues = (Map<String, Map<String, Double>>) in.readObject();
            in.close();
            fileIn.close();

            fileIn = new FileInputStream("predictedValues" + index + ".ser");
            in = new ObjectInputStream(fileIn);
            Map<String, Map<String, Double>> predictedValues = (Map<String, Map<String, Double>>) in.readObject();
            in.close();
            fileIn.close();

            int counter = 0;
            int totCount = 0;
            int numDiscarded = 0;

            for (String key : actualValues.keySet()) {
                Map<String, Double> actual = actualValues.get(key);
                Map<String, Double> predicted = predictedValues.get(key);

                actual = sortByValue(actual);
                predicted = sortByValue(predicted);

                if (actual.size() < 100) {
                    numDiscarded++;
                    continue;
                }

                List<String> actualClosest = new ArrayList<>(actual.keySet());
                actualClosest = actualClosest.subList(0, 250);

                List<String> predictedClosest = new ArrayList<>(predicted.keySet());
                Collections.shuffle(predictedClosest); //Uncomment to compare to random pick of sources
                predictedClosest = predictedClosest.subList(0, 100);

                for (int i = 0; i < 100; i++) {
                    if (predictedClosest.contains(actualClosest.get(i))) {
                        counter++;
                    }
                }
                System.out.println("Target " + key + ": " + counter + "/100, Total sources: " + predicted.size());
                totCount += counter;
                counter = 0;
            }
            totCount = totCount / predictedValues.keySet().size();

            System.out.println("\tClassifier n°" + index + ": " + totCount + "/100 avg, N° of discarded targets: " + numDiscarded);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new ArrayList<>(map.entrySet());
        list.sort(Map.Entry.comparingByValue());

        Map<K, V> result = new LinkedHashMap<>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }

        return result;
    }
}
