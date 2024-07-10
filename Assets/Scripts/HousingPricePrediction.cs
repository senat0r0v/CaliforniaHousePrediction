using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using TMPro; // Add this line to use TextMeshPro

public class HousingPricePrediction : MonoBehaviour
{
    public NNModel modelAsset;
    private Model model;
    private IWorker worker;

    // Serialized fields for user inputs using TMP_InputField
    public TMP_InputField medIncInput;
    public TMP_InputField houseAgeInput;
    public TMP_InputField aveRoomsInput;
    public TMP_InputField aveBedrmsInput;
    public TMP_InputField populationInput;
    public TMP_InputField aveOccupInput;
    public TMP_InputField latitudeInput;
    public TMP_InputField longitudeInput;

    // Serialized field for the result display
    public TMP_Text resultText;

    // Serialized field for the predict button
    public Button predictButton;

    void Start()
    {
        // Load the ONNX model
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // Add listener to the predict button
        predictButton.onClick.AddListener(Predict);
    }

    public void Predict()
    {
        // Retrieve user inputs
        float[] inputFeatures = new float[8];
        inputFeatures[0] = float.Parse(medIncInput.text);    // Median Income
        inputFeatures[1] = float.Parse(houseAgeInput.text);  // House Age
        inputFeatures[2] = float.Parse(aveRoomsInput.text);  // Average Rooms
        inputFeatures[3] = float.Parse(aveBedrmsInput.text); // Average Bedrooms
        inputFeatures[4] = float.Parse(populationInput.text);// Population
        inputFeatures[5] = float.Parse(aveOccupInput.text);  // Average Occupants
        inputFeatures[6] = float.Parse(latitudeInput.text);  // Latitude
        inputFeatures[7] = float.Parse(longitudeInput.text); // Longitude

        // Create a Tensor from the input features
        Tensor inputTensor = new Tensor(1, 8, inputFeatures);

        // Execute the model
        worker.Execute(inputTensor);
        
        // Get the output
        Tensor outputTensor = worker.PeekOutput();
        float predictedPrice = outputTensor[0];

        // Display the result
        resultText.text = "Predicted House Price: $" + predictedPrice.ToString("F2");

        // Release tensors
        inputTensor.Dispose();
        outputTensor.Dispose();
    }

    void OnDestroy()
    {
        // Dispose of the worker
        worker.Dispose();
    }
}
