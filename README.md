# User Case
wanted the ESP32-S3 to understand simple English commands like:

## Train Model
### Dataset
text,label
turn on the light,LIGHT_ON
switch off the light,LIGHT_OFF
hello,UNKNOWN

### Text → Numbers

Neural networks cannot understand words.
They only understand numbers.
We used a Keras tokenizer.
Suppose our vocabulary becomes

turn      -> 28
on        -> 15
light     -> 5
switch    -> 21
off       -> 13

switch on light
[21, 15, 5, 0, 0, 0]

switch = 21
on     = 15
light  = 5

ESP32 is : {21,15,5,0,0,0}

### Vocabulary
The tokenizer built a dictionary.
{
 "light":5,
 "switch":21,
 "off":13,
 ...
}

### Sequence Padding
max_len = 6

### Label Encoding
LIGHT_OFF -> 0
LIGHT_ON -> 1
UNKNOWN -> 2

### TensorFlow Lite Conversion
Microcontrollers cannot load Keras models, so convert it to .tflite which is 37 KB

### Convert Model to C Header 
This the model stored on micro controller - This is literally the TensorFlow Lite model stored as a byte array in flash memory

The ESP32 has no filesystem by default. light_model.tflite
convert light weight to tflite which is model.h file in this project

### Tokenizer on MC
ESP32 cannot execute Python. So we manually recreated
This "switch on lights" converts to "[21,15,44]" on MC

### TensorFlow Lite Micro
the ESP32 runs TensorFlow Lite Micro
It includes
 - interpreter
 - memory allocator
 - operators

Everything is optimized for embedded devices.

### Tensor Arena

The ESP32 doesn't allocate memory dynamically for every operation.

We create one block of RAM:
```
uint8_t tensor_arena[20 * 1024];
```

## Run infrence
```
Sentence -> Tokenizer -> Word IDs -> Padding ->  Input Tensor -> Interpreter -> Output Tensor
```

### Final 
LIGHT_ON ->  digitalWrite(GPIO,HIGH)

### Complete flow

```mermaid
flowchart TD
    %% Styling and Subgraphs
    subgraph Training [TRAINING - Jetson Orin Nano]
        A[CSV Dataset] --> B[Tokenizer <br><i>build vocabulary</i>]
        B --> C[Word IDs]
        C --> D[Padding <br><i>length = 6</i>]
        D --> E[Neural Network Training]
        E --> F[.keras Model]
        F --> G[TensorFlow Lite Converter]
        G --> H[light_model.tflite <br><i>37 KB</i>]
        H --> I[Convert to C header]
        I --> J([model.h])
    end

    subgraph Deployment [DEPLOYMENT - ESP32-S3]
        K[User Text] --> L[C++ Tokenizer]
        L --> M[6 Integer IDs]
        M --> N[TensorFlow Lite Micro]
        N --> O[Output Probabilities]
        O --> P[Highest Probability Intent]
        P --> Q[GPIO / LED / Relay / Wi-Fi Response]
    end

    %% Connect the two phases via the exported model header
    J -. Bridge Model .-> N

    %% Custom colors for GitHub Dark/Light themes
    style Training fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Deployment fill:#111827,stroke:#10b981,stroke-width:2px,color:#fff
    style J fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff
```