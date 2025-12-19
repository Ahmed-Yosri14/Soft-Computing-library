package NeuralNetwork.Data;

import java.io.*;
import java.util.*;

public class DataLoader {

    // Load CSV file from resources and return as list of string arrays
    public static List<String[]> loadCSV(String fileName) throws IOException {
        List<String[]> data = new ArrayList<>();

        // Get file from resources folder
        InputStream is = DataLoader.class
                .getClassLoader()
                .getResourceAsStream(fileName);

        if (is == null) {
            throw new FileNotFoundException("File not found in resources: " + fileName);
        }

        // Read file line by line
        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        String line;

        // Split each line by comma and add to data list
        while ((line = br.readLine()) != null) {
            data.add(line.split(","));
        }

        br.close();
        return data;
    }
}
