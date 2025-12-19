import FuzzyLogic.Apis.*;
import FuzzyLogic.Inference.*;
import FuzzyLogic.Rules.*;
import FuzzyLogic.Variable.*;
import GeneticAlgorithm.Chromosomes.Chromosome;
import GeneticAlgorithm.Fitness.*;
import GeneticAlgorithm.Selection.*;
import GeneticAlgorithm.Replacement.*;
import GeneticAlgorithm.GeneticAlgorithm;
import NeuralNetwork.Core.NeuralNetwork;
import NeuralNetwork.Layers.Dense;
import NeuralNetwork.Activations.*;
import NeuralNetwork.Loss.*;
import NeuralNetwork.init.*;
import NeuralNetwork.Optimizer.*;
import NeuralNetwork.Data.*;

import java.util.*;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("\n============================================================");
        System.out.println("        SOFT COMPUTING LIBRARY");
        System.out.println("============================================================");
        System.out.println("  Genetic Algorithms | Fuzzy Logic | Neural Networks");
        System.out.println("============================================================\n");

        boolean running = true;
        while (running) {
            System.out.println("\n============================================================");
            System.out.println("                    MAIN MENU");
            System.out.println("============================================================");
            System.out.println("  1. Genetic Algorithm (GA) - Optimization");
            System.out.println("  2. Fuzzy Logic (FL) - Approximate Reasoning");
            System.out.println("  3. Neural Network (NN) - Machine Learning");
            System.out.println("  4. Exit");
            System.out.println("============================================================");

            System.out.print("\nEnter your choice (1-4): ");
            int choice = 0;
            try {
                String input = scanner.nextLine().trim();
                if (!input.isEmpty()) {
                    choice = Integer.parseInt(input);
                }
            } catch (Exception e) {
                System.out.println("Invalid input. Please enter a number.");
                continue;
            }

            switch (choice) {
                case 1:
                    runGeneticAlgorithm(scanner);
                    break;
                case 2:
                    runFuzzyLogic(scanner);
                    break;
                case 3:
                    runNeuralNetwork(scanner);
                    break;
                case 4:
                    System.out.println("\n============================================================");
                    System.out.println("Thank you for using the Soft Computing Library!");
                    System.out.println("============================================================");
                    running = false;
                    break;
                default:
                    System.out.println("Invalid choice. Please enter 1-4.");
            }

            if (running) {
                System.out.println("\nPress Enter to return to main menu...");
                scanner.nextLine();
            }
        }

        scanner.close();
    }

    // ========================================================================
    //                    GENETIC ALGORITHM
    // ========================================================================

    private static void runGeneticAlgorithm(Scanner scanner) {
        System.out.println("\n============================================================");
        System.out.println("           GENETIC ALGORITHM - Order Delivery");
        System.out.println("============================================================");

        System.out.print("Number of delivery points (default 8): ");
        int numberOfDeliveryPoints = 8;
        String line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                int input = Integer.parseInt(line);
                if (input > 0) numberOfDeliveryPoints = input;
            } catch (Exception e) {
            }
        }

        System.out.print("Time constraint (default 120): ");
        int timeConstraint = 120;
        line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                int input = Integer.parseInt(line);
                if (input > 0) timeConstraint = input;
            } catch (Exception e) {
            }
        }

        System.out.print("Population size (default 50): ");
        int populationSize = 50;
        line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                int input = Integer.parseInt(line);
                if (input > 0) populationSize = input;
            } catch (Exception e) {
            }
        }

        System.out.print("Generations (default 100): ");
        int generations = 100;
        line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                int input = Integer.parseInt(line);
                if (input > 0) generations = input;
            } catch (Exception e) {
            }
        }

        System.out.print("Crossover rate 0-1 (default 0.7): ");
        double crossoverRate = 0.7;
        line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                double input = Double.parseDouble(line);
                if (input >= 0 && input <= 1.0) crossoverRate = input;
            } catch (Exception e) {
            }
        }

        System.out.print("Mutation rate 0-1 (default 0.02): ");
        double mutationRate = 0.02;
        line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                double input = Double.parseDouble(line);
                if (input >= 0 && input <= 1.0) mutationRate = input;
            } catch (Exception e) {
            }
        }

        // Selection method
        System.out.println("\nSelection Method:");
        System.out.println("1. Tournament Selection");
        System.out.println("2. Roulette Wheel Selection");
        System.out.print("Choose (1-2, default 1): ");
        int selectionChoice = 1;
        line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                int input = Integer.parseInt(line);
                if (input == 1 || input == 2) selectionChoice = input;
            } catch (Exception e) {
            }
        }

        Selection selectionMethod;
        if (selectionChoice == 1) {
            System.out.print("Tournament size (default 3): ");
            int tournamentSize = 3;
            line = scanner.nextLine().trim();
            if (!line.isEmpty()) {
                try {
                    int input = Integer.parseInt(line);
                    if (input > 0) tournamentSize = input;
                } catch (Exception e) {
                }
            }
            selectionMethod = new TournamentSelection(tournamentSize);
        } else {
            selectionMethod = new RouletteWheelSelection();
        }

        // Replacement strategy
        System.out.println("\nReplacement Strategy:");
        System.out.println("1. Generational (Complete replacement)");
        System.out.println("2. Steady-State (Replace K parents)");
        System.out.println("3. Elitist (Keep best individuals)");
        System.out.print("Choose (1-3, default 3): ");
        int replacementChoice = 3;
        line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                int input = Integer.parseInt(line);
                if (input >= 1 && input <= 3) replacementChoice = input;
            } catch (Exception e) {
            }
        }

        ReplacementStrategy replacementStrategy;
        if (replacementChoice == 1) {
            replacementStrategy = new GenerationalReplacement();
        } else if (replacementChoice == 2) {
            System.out.print("Number of parents to replace (default 2): ");
            int k = 2;
            line = scanner.nextLine().trim();
            if (!line.isEmpty()) {
                try {
                    int input = Integer.parseInt(line);
                    if (input > 0) k = input;
                } catch (Exception e) {
                }
            }
            replacementStrategy = new SteadyStateReplacement(k);
        } else {
            System.out.print("Number of elite individuals (default 2): ");
            int eliteCount = 2;
            line = scanner.nextLine().trim();
            if (!line.isEmpty()) {
                try {
                    int input = Integer.parseInt(line);
                    if (input > 0) eliteCount = input;
                } catch (Exception e) {
                }
            }
            replacementStrategy = new ElitistReplacement(eliteCount);
        }

        // Generate distance matrix
        ArrayList<ArrayList<Integer>> distanceMatrix = new ArrayList<>();
        Random rand = new Random(42);
        int n = numberOfDeliveryPoints + 1;
        for (int i = 0; i < n; i++) {
            distanceMatrix.add(new ArrayList<>());
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    distanceMatrix.get(i).add(0);
                } else if (j < i) {
                    distanceMatrix.get(i).add(distanceMatrix.get(j).get(i));
                } else {
                    distanceMatrix.get(i).add(rand.nextInt(40) + 10);
                }
            }
        }

        // Setup and run GA
        FitnessEvaluator.getInstance(distanceMatrix, timeConstraint);
        GeneticAlgorithm ga = new GeneticAlgorithm();
        ga.setPopulationSize(populationSize);
        ga.setChromosomeLength(numberOfDeliveryPoints);
        ga.setFitnessFunction(FitnessEvaluator.getInstance());
        ga.setChromosomeType(GeneticAlgorithm.ChromosomeType.INTEGER);
        ga.setCrossoverRate(crossoverRate);
        ga.setMutationRate(mutationRate);
        ga.setGenerations(generations);
        ga.setSelectionMethod(selectionMethod);
        ga.setReplacementStrategy(replacementStrategy);

        System.out.println("\nRunning GA...\n");
        ga.run();

        Chromosome best = ga.getBestSolution();
        System.out.println("\n============================================================");
        System.out.println("                    RESULTS");
        System.out.println("============================================================");
        System.out.println("Best Sequence: " + best.getDeliverySequence());
        System.out.println("Orders Delivered: " + best.getFitness());
        System.out.println("Total Time: " + best.getTotalRouteTime() + " units");
        System.out.println("Time Constraint: " + timeConstraint + " units");
        System.out.println("============================================================");

        ga.printStatistics();
    }

    // ========================================================================
    //                    FUZZY LOGIC
    // ========================================================================

    private static void runFuzzyLogic(Scanner scanner) {
        System.out.println("\n============================================================");
        System.out.println("           FUZZY LOGIC - Smart Irrigation");
        System.out.println("============================================================");

        // Setup variables
        SoilMoisture soil = new SoilMoisture();
        Temperature temp = new Temperature();
        RainForecast rain = new RainForecast();
        WaterDuration water = new WaterDuration();

        Map<String, FuzzyVariable> inputs = new HashMap<>();
        inputs.put(soil.getName(), soil);
        inputs.put(temp.getName(), temp);
        inputs.put(rain.getName(), rain);
        inputs.put(water.getName(), water);

        // Load rules
        SimpleRuleManager ruleManager = new SimpleRuleManager();
        List<RuleDocument> enabledRules = ruleManager.loadEnabledRules();
        System.out.println("Loaded " + enabledRules.size() + " rules from rules.json");

        // Convert rules
        List<FuzzyRule> fuzzyRules = new ArrayList<>();
        List<FuzzyRule> sugenoRules = new ArrayList<>();
        double outputMin = 0.0;
        double outputMax = 30.0;

        for (RuleDocument doc : enabledRules) {
            fuzzyRules.add(RuleConverter.toFuzzyRule(doc, inputs));
            sugenoRules.add(RuleConverter.toSugenoRule(doc, inputs, outputMin, outputMax));
        }

        // Scenario selection
        System.out.println("\n1. Custom input values");
        System.out.println("2. Predefined scenarios");
        System.out.print("Choose (1-2, default 2): ");
        int scenarioChoice = 2;
        String line = scanner.nextLine().trim();
        if (!line.isEmpty()) {
            try {
                int input = Integer.parseInt(line);
                if (input == 1 || input == 2) scenarioChoice = input;
            } catch (Exception e) {
            }
        }

        if (scenarioChoice == 1) {
            // Custom scenario
            System.out.print("\nSoil Moisture % (default 50): ");
            double soilValue = 50;
            line = scanner.nextLine().trim();
            if (!line.isEmpty()) {
                try {
                    double input = Double.parseDouble(line);
                    if (input >= 0) soilValue = input;
                } catch (Exception e) {
                }
            }

            System.out.print("Temperature °C (default 25): ");
            double tempValue = 25;
            line = scanner.nextLine().trim();
            if (!line.isEmpty()) {
                try {
                    double input = Double.parseDouble(line);
                    if (input >= 0) tempValue = input;
                } catch (Exception e) {
                }
            }

            System.out.print("Rain Forecast mm (default 5): ");
            double rainValue = 5;
            line = scanner.nextLine().trim();
            if (!line.isEmpty()) {
                try {
                    double input = Double.parseDouble(line);
                    if (input >= 0) rainValue = input;
                } catch (Exception e) {
                }
            }

            processScenario("Custom Scenario", soilValue, tempValue, rainValue,
                    soil, temp, rain, inputs, fuzzyRules, sugenoRules, outputMin, outputMax);

        } else {
            // Predefined scenarios
            double[][] scenarios = {
                    {80.0, 24.0, 18.0}, // High Moisture & Heavy Rain
                    {10.0, 34.0, 0.0},  // Low Moisture & No Rain
                    {35.0, 30.0, 3.0}   // Moderate
            };
            String[] names = {"High Moisture & Heavy Rain", "Low Moisture & No Rain", "Moderate Conditions"};

            for (int i = 0; i < scenarios.length; i++) {
                processScenario(names[i], scenarios[i][0], scenarios[i][1], scenarios[i][2],
                        soil, temp, rain, inputs, fuzzyRules, sugenoRules, outputMin, outputMax);
            }
        }

        System.out.println("\n============================================================");
        System.out.println("       FUZZY LOGIC EXECUTION COMPLETED");
        System.out.println("============================================================");
    }

    private static void processScenario(String name, double soilValue, double tempValue, double rainValue,
                                        SoilMoisture soil, Temperature temp, RainForecast rain,
                                        Map<String, FuzzyVariable> inputs, List<FuzzyRule> fuzzyRules,
                                        List<FuzzyRule> sugenoRules, double outputMin, double outputMax) {
        System.out.println("\n------------------------------------------------------------");
        System.out.println("SCENARIO: " + name);
        System.out.println("------------------------------------------------------------");

        soil.setValue(soilValue);
        temp.setValue(tempValue);
        rain.setValue(rainValue);

        System.out.println("Inputs: Soil=" + soilValue + "%, Temp=" + tempValue + "°C, Rain=" + rainValue + "mm");

        MamdaniInferenceEngine mamdani = new MamdaniInferenceEngine(inputs, fuzzyRules, outputMin, outputMax);
        double mamdaniResult = mamdani.evaluate();

        SugenoInferenceEngine sugeno = new SugenoInferenceEngine(inputs, sugenoRules);
        double sugenoResult = sugeno.evaluate();

        System.out.println("Mamdani: " + String.format("%.1f", mamdaniResult) + " minutes");
        System.out.println("Sugeno:  " + String.format("%.1f", sugenoResult) + " minutes");

        if (mamdaniResult < 8) {
            System.out.println("Recommendation: SHORT watering");
        } else if (mamdaniResult < 18) {
            System.out.println("Recommendation: MEDIUM watering");
        } else {
            System.out.println("Recommendation: LONG watering");
        }
    }

    // ========================================================================
    //                    NEURAL NETWORK
    // ========================================================================

    private static void runNeuralNetwork(Scanner scanner) {
        System.out.println("\n============================================================");
        System.out.println("           NEURAL NETWORK - Classification");
        System.out.println("============================================================");

        try {
            // Load data
            List<String[]> rawData = DataLoader.loadCSV("column_2C.csv");
            int samples = rawData.size() - 1;
            double[][] X = new double[samples][6];
            double[][] y = new double[samples][1];

            for (int i = 0; i < samples; i++) {
                String[] row = rawData.get(i + 1);
                for (int j = 0; j < 6; j++) {
                    try {
                        X[i][j] = Double.parseDouble(row[j]);
                    } catch (Exception e) {
                        X[i][j] = 0.0;
                    }
                }
                y[i][0] = row[6].equalsIgnoreCase("Abnormal") ? 1.0 : 0.0;
            }

            System.out.println("Dataset loaded: " + samples + " samples");

            // Split and normalize
            SplitResult split = DataSplit.split(X, y, 0.8);
            var stats = NormalizationUtils.fitMinMax(split.Xtrain());
            NormalizationUtils.transformMinMax(split.Xtrain(), stats);
            NormalizationUtils.transformMinMax(split.Xtest(), stats);

            System.out.println("Train: " + split.Xtrain().length + ", Test: " + split.Xtest().length);

            // Configuration
            System.out.println("\n1. Default configuration");
            System.out.println("2. Custom configuration");
            System.out.print("Choose (1-2, default 1): ");
            int configChoice = 1;
            String line = scanner.nextLine().trim();
            if (!line.isEmpty()) {
                try {
                    int input = Integer.parseInt(line);
                    if (input == 1 || input == 2) configChoice = input;
                } catch (Exception e) {
                }
            }

            int epochs = 500;
            int batchSize = 8;
            double learningRate = 0.1;

            if (configChoice == 2) {
                System.out.print("Epochs (default 500): ");
                line = scanner.nextLine().trim();
                if (!line.isEmpty()) {
                    try {
                        int input = Integer.parseInt(line);
                        if (input > 0) epochs = input;
                    } catch (Exception e) {
                    }
                }

                System.out.print("Batch size (default 8): ");
                line = scanner.nextLine().trim();
                if (!line.isEmpty()) {
                    try {
                        int input = Integer.parseInt(line);
                        if (input > 0) batchSize = input;
                    } catch (Exception e) {
                    }
                }

                System.out.print("Learning rate (default 0.1): ");
                line = scanner.nextLine().trim();
                if (!line.isEmpty()) {
                    try {
                        double input = Double.parseDouble(line);
                        if (input > 0) learningRate = input;
                    } catch (Exception e) {
                    }
                }
            }

            // Build network
            NeuralNetwork nn = new NeuralNetwork();
            nn.addLayer(new Dense(6, 12, new Xavier()));
            nn.addLayer(new ReLU());
            nn.addLayer(new Dense(12, 6, new Xavier()));
            nn.addLayer(new ReLU());
            nn.addLayer(new Dense(6, 1, new Xavier()));
            nn.addLayer(new Sigmoid());
            nn.setOptimizer(new SGD(learningRate));

            // Train
            System.out.println("\nTraining...");
            nn.train(split.Xtrain(), split.ytrain(), new BinaryCrossEntropy(), epochs, batchSize);

            // Evaluate
            double accuracy = nn.evaluate(split.Xtest(), split.ytest());
            System.out.println("\n============================================================");
            System.out.println("Test Accuracy: " + String.format("%.2f%%", accuracy * 100));
            System.out.println("============================================================");

            // Sample predictions
            System.out.println("\nSample predictions:");
            for (int i = 0; i < Math.min(5, split.Xtest().length); i++) {
                double p = nn.predict(split.Xtest()[i])[0];
                String predicted = p >= 0.5 ? "Abnormal" : "Normal";
                String actual = split.ytest()[i][0] == 1.0 ? "Abnormal" : "Normal";
                System.out.println("  Sample " + (i + 1) + ": " + predicted + " (actual: " + actual + ")");
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
