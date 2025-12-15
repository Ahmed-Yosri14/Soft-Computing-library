package NeuralNetwork.Data;

import java.io.*;
import java.util.*;

public class DataLoader {

    public static List<String[]> loadCSV(String fileName) throws IOException {
        List<String[]> data = new ArrayList<>();

        InputStream is = DataLoader.class
                .getClassLoader()
                .getResourceAsStream(fileName);

        if (is == null) {
            throw new FileNotFoundException("File not found in resources: " + fileName);
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        String line;

        while ((line = br.readLine()) != null) {
            data.add(line.split(","));
        }

        br.close();
        return data;
    }
}
