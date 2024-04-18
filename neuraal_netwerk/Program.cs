using System;
using System.Linq;

public class EenvoudigNeuraalNetwerk
{
    // Gewichten van input naar verborgen laag
    private double[,] inputNaarVerborgenGewichten;
    // Gewichten van verborgen laag naar output
    private double[] verborgenNaarOutputGewichten;
    // Neuronen in de verborgen laag
    private double[] verborgenLaag;
    // Uitvoer van het netwerk
    private double uitvoer;

    // Constructor voor het initialiseren van het netwerk
    public EenvoudigNeuraalNetwerk(int verborgenNeuronen)
    {
        // Initialiseren van de gewichten en lagen
        this.inputNaarVerborgenGewichten = new double[4, verborgenNeuronen];
        this.verborgenNaarOutputGewichten = new double[verborgenNeuronen];
        this.verborgenLaag = new double[verborgenNeuronen];

        // Gewichten initialiseren met willekeurige waarden
        InitialiseerGewichten();
    }

    // Functie om de gewichten te initialiseren
    private void InitialiseerGewichten()
    {
        Random willekeurig = new Random();

        // Gewichten van input naar verborgen laag initialiseren
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < inputNaarVerborgenGewichten.GetLength(1); j++)
            {
                inputNaarVerborgenGewichten[i, j] = willekeurig.NextDouble();
            }
        }

        // Gewichten van verborgen laag naar output initialiseren
        for (int j = 0; j < verborgenNaarOutputGewichten.Length; j++)
        {
            verborgenNaarOutputGewichten[j] = willekeurig.NextDouble();
        }
    }

    // Sigmoid activatiefunctie
    private double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    // Feedforward methode om de uitvoer te berekenen
    public double FeedForward(double[] input)
    {
        // Bereken de waarden voor de verborgen laag
        for (int j = 0; j < verborgenLaag.Length; j++)
        {
            double som = 0;
            for (int i = 0; i < input.Length; i++)
            {
                som += input[i] * inputNaarVerborgenGewichten[i, j];
            }
            verborgenLaag[j] = Sigmoid(som);
        }

        // Bereken de uitvoer
        double somUitvoer = 0;
        for (int j = 0; j < verborgenLaag.Length; j++)
        {
            somUitvoer += verborgenLaag[j] * verborgenNaarOutputGewichten[j];
        }
        uitvoer = Sigmoid(somUitvoer);

        return uitvoer;
    }

    // Trainingsmethode om het netwerk te trainen
    public void Train(double[] input, double doelUitvoer, double leersnelheid)
    {
        double voorspeldeUitvoer = FeedForward(input);
        double fout = doelUitvoer - voorspeldeUitvoer;

        // Update de gewichten van verborgen laag naar output
        for (int j = 0; j < verborgenNaarOutputGewichten.Length; j++)
        {
            verborgenNaarOutputGewichten[j] += fout * verborgenLaag[j] * leersnelheid;
        }

        // Update de gewichten van input naar verborgen laag
        for (int i = 0; i < input.Length; i++)
        {
            for (int j = 0; j < verborgenLaag.Length; j++)
            {
                inputNaarVerborgenGewichten[i, j] += fout * input[i] * verborgenLaag[j] * (1 - verborgenLaag[j]) * leersnelheid;
            }
        }
    }

    // Hoofdprogramma om het netwerk te testen
    public static void Main()
    {
        EenvoudigNeuraalNetwerk nn = new EenvoudigNeuraalNetwerk(3);

        // Training data en labels
        double[][] trainingData =
        {
            new double[] { 0, 0, 1, 0 },
            new double[] { 0, 1, 1, 1 },
            new double[] { 1, 0, 1, 1 },
            new double[] { 1, 1, 1, 0 }
        };

        double[] trainingLabels = { 0, 1, 1, 0 };

        double leersnelheid = 6;
        int epochs = 20000;

        // Training van het netwerk
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < trainingData.Length; i++)
            {
                nn.Train(trainingData[i].Take(trainingData[i].Length - 1).ToArray(), trainingLabels[i], leersnelheid);
            }
        }
        // Testen van het getrainde netwerk met vier verschillende testinputs
        double[][] testInputs =
        {
            new double[] { 1, 0, 1 },
            new double[] { 0, 1, 0 },
            new double[] { 1, 1, 0 },
            new double[] { 0, 0, 1 },
        };
        double totaleFout = 0;

        for (int i = 0; i < testInputs.Length; i++)
        {
            double voorspeldeUitvoer = nn.FeedForward(testInputs[i]);
            double fout = Math.Abs(trainingLabels[i] - voorspeldeUitvoer);
            totaleFout += fout;

            // Converteer de testinput array naar een leesbare string
            string testInputString = string.Join(", ", testInputs[i]);
            
            Console.WriteLine($"Test input: [{testInputString}]    Voorspelde uitvoer: {voorspeldeUitvoer}    Fout: {fout}");
        }
        double gemiddeldeFout = totaleFout / testInputs.Length;
        Console.WriteLine($"Gemiddelde fout: {gemiddeldeFout}");
    }
}