
import NeuralNetwork.CaseStudyDemo;
import FuzzyLogic.FuzzyLogicCaseStudyDemo;
import GeneticAlgorithm.CaseStudy;
import java.util.Scanner;

public class Main {

    private static int readChoice(Scanner sc, int min, int max, int def) {
        try {
            System.out.print("Choose option: ");
            int choice = Integer.parseInt(sc.nextLine());
            if (choice < min || choice > max) {
                System.out.println("Invalid choice. Using default: " + def);
                return def;
            }
            return choice;
        } catch (Exception e) {
            System.out.println("Invalid input. Using default: " + def);
            return def;
        }
    }

    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        System.out.println("========================================");
        System.out.println("      ARTIFICIAL INTELLIGENCE DEMOS     ");
        System.out.println("========================================");
        System.out.println("1) Neural Network – Vertebral Column Classification");
        System.out.println("2) Fuzzy Logic – Irrigation System");
        System.out.println("3) Genetic Algorithm – Optimization Demo");
        System.out.println("0) Exit");
        System.out.println("========================================");

        int choice = readChoice(scanner, 0, 3, 0);
        System.out.println();

        switch (choice) {
            case 1 -> {
                System.out.println("Launching Neural Network Demo...\n");
                CaseStudyDemo.main(new String[]{});
            }
            case 2 -> {
                System.out.println("Launching Fuzzy Logic Demo...\n");
                FuzzyLogicCaseStudyDemo.main(new String[]{});
            }
            case 3 -> {
                System.out.println("Launching Genetic Algorithm Demo...\n");
                CaseStudy.main(new String[]{});
            }
            case 0 -> {
                System.out.println("Exiting system. Goodbye!");
            }
        }

        scanner.close();
    }
}
