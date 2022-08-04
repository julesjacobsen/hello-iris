package org.monarchinitiative.exomiser.helloiris;

import ai.onnxruntime.*;
import org.springframework.stereotype.Component;

import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Map;

@Component
public class OnnxScorer {

    private static final String MICRO = Character.toString(0x03BC);
    // https://github.com/microsoft/onnxruntime/discussions/10107
    // environment and session are thread-safe
    private final OrtEnvironment env;
    private final OrtSession session;

    // super-simple demo of reading in an onnx model of a random forest trained on the iris dataset and using the onnx
    // runtime in Java to run some inferences
    public OnnxScorer() throws OrtException {
        Path modelPath = Path.of("models/rf-iris.onnx");
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions());
        System.out.println(session.getInputInfo());
        System.out.println(session.getInputNames()); // use this as the input names for the session otherwise... boom!

        System.out.println(session.getNumOutputs()); // size of the results
        System.out.println(session.getOutputInfo()); // what to expect from the result when session.run is called

        // These inputs and outputs all differ depending on the model and how it was trained.
        // In this case the rf-iris.onnx (trained in variant_classifier/hello_ml_world.py) is trained on 4 float values
        // from the iris dataset and returns the integer label [0, 1, 2] with a corresponding probability depending on
        // the classification e.g 0 ='setosa', 1 ='versicolor', 2='virginica'
        OnnxTensor onnxTensor = OnnxTensor.createTensor(env, new float[][]{{5.1f, 3.5f, 1.4f, 0.2f}, {5.9f, 3.0f, 4.2f, 1.5f}, {5.9f, 3.0f, 5.1f, 1.8f}});

        // requires the name(s) from session.getInputNames() - these are set when the model is exported.
        var inputs = Map.of("float_input", onnxTensor);
        Instant start = Instant.now();
        try (var results = session.run(inputs)) {
            Instant end = Instant.now();
            System.out.println("Inference took " + Duration.between(start, end).toNanos() / 1000 + " " + MICRO + "s");
            // Only iterates once
            for (Map.Entry<String, OnnxValue> result : results) {
                OnnxValue resultValue = result.getValue();
                System.out.println(result.getKey() + ": " + resultValue);
                if (resultValue instanceof OnnxTensor resultTensor) {
                    var valueType = resultTensor.getType();
                    var info = resultTensor.getInfo();
                    System.out.println(resultTensor.getInfo().isScalar());
                    long[] labels = (long[]) resultTensor.getValue();
                    System.out.println("label: " + valueType + " " + info + " lables: " + Arrays.stream(labels).boxed().toList());
                }
                if (resultValue instanceof OnnxSequence onnxSequence) {
                    if (onnxSequence.getInfo().isSequenceOfMaps()) {
                        System.out.println("result is a sequence of maps");
                    }
                    // returns [{0=0.99999934, 1=0.0, 2=0.0}, {0=0.0, 1=0.99999934, 2=0.0}, {0=0.0, 1=0.0, 2=0.99999934}]
                    // where 0, 1, 2 are the labels predicted
                    System.out.println("p=" + onnxSequence.getValue());
                }
            }
        }
    }
}
