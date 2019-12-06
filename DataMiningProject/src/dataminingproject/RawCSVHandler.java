/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataminingproject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

/**
 *
 * @author Lorenzo
 */
public class RawCSVHandler {

    private String path;

    public RawCSVHandler(String path) {
        try {
            this.path = path;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //Returns a set containing all the destination ids present in the dataset
    public Set<Integer> getUniqueDestinationID() {
        Set<Integer> uniqueValues = new HashSet<Integer>();

        String row;
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(path));
            String names = csvReader.readLine();
            while ((row = csvReader.readLine()) != null) {
                String[] data = row.split(",");
                uniqueValues.add(Integer.parseInt(data[8])); //Destination ID
            }
            csvReader.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
        return uniqueValues;
    }

    //Creates the test and train set for the 
    public void splitTrainTestSet() {
        System.out.println("Start splitting test and training set...");

        ArrayList<Integer> uv = new ArrayList<>(getUniqueDestinationID());
        Collections.sort(uv);
        Collections.shuffle(uv, new Random(Parameters.RANDOM_SEED));
        int testSize = uv.size() / Parameters.K_FOLD;

        for (int i = 0; i < Parameters.K_FOLD; i++) {

            List<Integer> testIndexes = uv.subList(i * testSize, i * testSize + testSize - 1);
            List<Integer> trainIndexes = new ArrayList<>(uv);
            trainIndexes.removeAll(testIndexes);

            Collections.sort(testIndexes);
            Collections.sort(trainIndexes);

            System.out.println(testIndexes);

            String row;
            try (BufferedReader csvReader = new BufferedReader(new FileReader(path));
                    FileWriter csvTest = new FileWriter("traintest_sets\\test" + i + ".csv");
                    FileWriter csvTrain = new FileWriter("traintest_sets\\train" + i + ".csv");) {
                String names = csvReader.readLine();
                names.split(",");
                csvTest.append(String.join(",", names));
                csvTest.append("\n");
                csvTrain.append(String.join(",", names));
                csvTrain.append("\n");
                while ((row = csvReader.readLine()) != null) {
                    String[] data = row.split(",");
                    data[24] = String.valueOf(Double.parseDouble(data[24]) / (Double.parseDouble(data[22]) + Double.parseDouble(data[23])));
                    data[28] = String.valueOf(Double.parseDouble(data[28]) / (Double.parseDouble(data[26]) + Double.parseDouble(data[27])));
                    data[32] = String.valueOf(Double.parseDouble(data[32]) / (Double.parseDouble(data[30]) + Double.parseDouble(data[31])));

                    if (testIndexes.contains(Integer.parseInt(data[8]))) {
                        csvTest.append(String.join(",", data));
                        csvTest.append("\n");
                    } else if (trainIndexes.contains(Integer.parseInt(data[8]))) {
                        csvTrain.append(String.join(",", data));
                        csvTrain.append("\n");
                    }
                }
                csvTest.flush();
                csvTest.close();
                csvTrain.flush();
                csvTrain.close();
                csvReader.close();

            } catch (Exception e) {
                System.err.println("ERROR DURING THE SPLIT OF THE DATASET");
            }

        }

        System.out.println("DONE");
    }

    //Performes the preprocess of the complete dataset.
    public void preprocessComplete() {
        System.out.println("Starting preprocessing of the datasets...");
        List<String> result = null;
        try (Stream<Path> walk = Files.walk(Paths.get("traintest_sets\\"))) {

            result = walk.filter(Files::isRegularFile)
                    .map(x -> x.toString()).collect(Collectors.toList());

            for (String filePath : result) {
                applyFiltersToCompleteData(filePath);
            }

        } catch (IOException e) {
            System.err.println("ERROR DURING THE PREPROCESSING");
        }

        System.out.println("DONE");
    }

    //It performs the preprocessing of the reduced dataset
    public void preprocess() {
        System.out.println("Starting preprocessing of the datasets...");
        List<String> result = null;
        try (Stream<Path> walk = Files.walk(Paths.get("traintest_sets\\"))) {

            result = walk.filter(Files::isRegularFile)
                    .map(x -> x.toString()).collect(Collectors.toList());

            for (String filePath : result) {
                applyFilters(filePath);
                modifyHeader(filePath);
            }

        } catch (IOException e) {
            System.err.println("ERROR DURING THE PREPROCESSING");
        }

        System.out.println("DONE");
    }

    //This function copies the same values of possible nominal attribute in the arff header.
    //This is done because the possible values for a nominal attribute are automatically generated
    //when an arff file is saved, but it only consider the values present in the file.
    //But a certain attribute might have values that are not present in a certain file, so to make it
    //homogeneous, the header of the arff files has been uniformed, copying all the possible values
    //in all the headers.
    public void modifyHeader(String filePath) {
        filePath = (filePath.split("\\."))[0] + ".arff";
        String fileName = filePath.split("\\\\")[1];
        List<String> newHeader = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader("data\\complete_nominal_values.txt"));
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
                newHeader.add(line);
            }
            br.close();

            br = new BufferedReader(new FileReader(filePath));

            int lineCounter = 1;
            BufferedWriter bw = new BufferedWriter(new FileWriter("traintest_sets\\temp.arff"));
            while ((line = br.readLine()) != null) {
                if (lineCounter == 5) {
                    bw.write(newHeader.get(0));
                } else if (lineCounter == 6) {
                    bw.write(newHeader.get(1));
                } else if (lineCounter == 11) {
                    bw.write(newHeader.get(2));
                } else {
                    bw.write(line);
                }
                bw.newLine();
                lineCounter++;
            }
            br.close();
            bw.close();

            File f = new File(filePath);
            f.delete();

            new File("traintest_sets\\temp.arff").renameTo(f);

        } catch (Exception e) {
            System.out.println("ERROR");
        }
    }

    //Used for the reduced set and to predict RTT
    public void applyFilters(String filePath) {
        Instances data = null;
        try {
            DataSource source = new DataSource(filePath);
            data = source.getDataSet();
        } catch (Exception e) {
            System.err.println("ERROR DURING THE LOADING OF THE DATASET IN " + filePath);
        }
        if (data == null) {
            return;
        }

        try {
            data.setClassIndex(19);

            String[] opt = new String[]{"-R", "3,10,12-16,18-19"};
            Remove remove = new Remove();
            remove.setOptions(opt);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);

            opt = new String[]{"-R", "2,3,8,9,10"};
            NumericToNominal ntn = new NumericToNominal();
            ntn.setOptions(opt);
            ntn.setInputFormat(data);
            data = Filter.useFilter(data, ntn);

            Normalize norm = new Normalize();
            norm.setInputFormat(data);
            data = Filter.useFilter(data, norm);

            Reorder reorder = new Reorder();
            opt = new String[]{"-R", "1-10,12-last,11"};
            reorder.setOptions(opt);
            reorder.setInputFormat(data);
            data = Filter.useFilter(data, reorder);

            filePath = (filePath.split("\\."))[0] + ".arff";
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(filePath));
            saver.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //Used for complete set and to predict Geodetic distance
    public void applyFiltersToCompleteData(String filePath) {
        Instances data = null;
        try {
            DataSource source = new DataSource(filePath);
            data = source.getDataSet();
        } catch (Exception e) {
            System.err.println("ERROR DURING THE LOADING OF THE DATASET IN " + filePath);
        }
        if (data == null) {
            return;
        }

        try {
            String[] opt = new String[]{"-R", "3,10,12-16,18, 20"}; //Add 19 to remove geodesic distance
            Remove remove = new Remove();
            remove.setOptions(opt);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);

            opt = new String[]{"-R", "2,3,8,9,10"};
            NumericToNominal ntn = new NumericToNominal();
            ntn.setOptions(opt);
            ntn.setInputFormat(data);
            data = Filter.useFilter(data, ntn);

            Normalize norm = new Normalize();
            norm.setInputFormat(data);
            data = Filter.useFilter(data, norm);

            Reorder reorder = new Reorder();
            opt = new String[]{"-R", "1-10,12-last,11"};
            reorder.setOptions(opt);
            reorder.setInputFormat(data);
            data = Filter.useFilter(data, reorder);

            filePath = (filePath.split("\\."))[0] + ".arff";
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(filePath));
            saver.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
