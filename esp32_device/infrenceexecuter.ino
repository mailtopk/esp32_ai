#include <Arduino.h>
#include "TensorFlowLite_ESP32.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "tokenizer_vocab.h"
#include "labels.h"

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// Model is ~67 KB, but the model itself lives in flash.
// The tensor arena is RAM used by the interpreter.
//
// Start with 64 KB.
constexpr size_t kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Model configuration
constexpr int kSequenceLength = 8;
constexpr int kNumberOfClasses = 3;

// Tokenizer configuration
constexpr int kUnknownTokenId = 1;

// Utility: vocabulary lookup
int32_t FindTokenId(const char* word) {
    for (size_t i = 0; i < kVocabularySize; ++i) {
        if (strcmp(word, kVocabulary[i].word) == 0) {
            return kVocabulary[i].id;
        }
    }
    return kUnknownTokenId;
}

// ------------------------------------------------------------
// Simple tokenizer
// ------------------------------------------------------------
//   - converts A-Z to lowercase
//   - separates words on spaces
//   - removes common punctuation
//   - looks up vocabulary IDs
//   - pads to 8 tokens with 0
//
// ------------------------------------------------------------

void Tokenize( const char* text, int32_t* output_tokens) 
{

    // Padding ID is 0.
    for (int i = 0; i < kSequenceLength; ++i) {
        output_tokens[i] = 0;
    }

    char buffer[128];
    strncpy(buffer, text, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    // Convert to lowercase and replace punctuation with spaces.
    for (size_t i = 0; buffer[i] != '\0'; ++i) {
        char c = buffer[i];
        if (c >= 'A' && c <= 'Z') {
            buffer[i] = c + ('a' - 'A');
            continue;
        }
        switch (c) {
            case '.':
            case ',':
            case '!':
            case '?':
            case ';':
            case ':':
            case '"':
            case '\'':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
                buffer[i] = ' ';
                break;
            default:
                break;
        }
    }

    // Split into words.
    int token_index = 0;
    char* token = strtok(buffer, " \t\r\n");
    while (
        token != nullptr &&
        token_index < kSequenceLength
    ) {
        int32_t id = FindTokenId(token);
        output_tokens[token_index] = id;
        token_index++;
        token = strtok(nullptr, " \t\r\n");
    }
}

// Run inference
void RunInference(const char* text) {
    int32_t tokens[kSequenceLength];
    Tokenize(text, tokens);
    Serial.println();
    Serial.println("========================================");
    Serial.print("Text: ");
    Serial.println(text);

    Serial.print("Tokens: [");
    for (int i = 0; i < kSequenceLength; ++i) {
        Serial.print(tokens[i]);
        if (i < kSequenceLength - 1) {
            Serial.print(", ");
        }
    }
    Serial.println("]");

    // Copy tokens into TFLite input 
    for (int i = 0; i < kSequenceLength; ++i) {
        input->data.i32[i] = tokens[i];
    }

    // Invoke model
    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        Serial.println("ERROR: interpreter->Invoke() failed");
        return;
    }

    // Read output
    Serial.println("Output:");

    int best_class = 0;
    float best_probability = output->data.f[0];

    for (int i = 0; i < kNumberOfClasses; ++i) {
        float probability = output->data.f[i];

        Serial.print("  ");
        Serial.print(i);
        Serial.print(" ");
        Serial.print(kLabels[i]);
        Serial.print(": ");
        Serial.println(probability, 6);

        if (probability > best_probability) {
            best_probability = probability;
            best_class = i;
        }
    }

    Serial.print("Predicted class: ");
    Serial.print(best_class);

    Serial.print(" (");
    Serial.print(kLabels[best_class]);
    Serial.println(")");

    Serial.println("========================================");
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println("========================================");
    Serial.println("ESP32 TensorFlow Lite Micro");
    Serial.println("Text classification inference");
    Serial.println("========================================");

    // Error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    // Load model
    model = tflite::GetModel(light_model_tflite);
    if (model == nullptr) {
        Serial.println("ERROR: GetModel() returned nullptr");
        return;
    }

    Serial.print("Model schema version: ");
    Serial.println(model->version());

    Serial.print("Expected schema version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);


    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch");
        return;
    }

    // Resolver
    static tflite::AllOpsResolver resolver;
    // Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena,
        kTensorArenaSize, error_reporter );

    interpreter = &static_interpreter;
    TfLiteStatus allocate_status = interpreter->AllocateTensors();

    if (allocate_status != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);


    Serial.println();
    Serial.println("Model initialized successfully.");

    Serial.print("Input type: ");
    Serial.println(input->type);

    Serial.print("Input bytes: ");
    Serial.println(input->bytes);

    Serial.print("Output type: ");
    Serial.println(output->type);

    Serial.print("Output bytes: ");
    Serial.println(output->bytes);


    // --------------------------------------------------------
    // Run our known test cases
    RunInference("turn the light on");
    RunInference("turn the light off");
    RunInference("please turn the bedroom light on";
    RunInference("hello");
    RunInference( "make the room brighter");

    Serial.println();
    Serial.println("Type a sentence and press ENTER.");
    Serial.println();
}

void loop() {

    static char input_buffer[128];
    static size_t input_length = 0;

    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r' || c == '\n') {
            if (input_length > 0) {
                input_buffer[input_length] = '\0';
                RunInference(input_buffer);
                input_length = 0;
            }
        } else {
            if ( input_length < sizeof(input_buffer) - 1) {
                input_buffer[input_length++] = c;
            }
        }
    }
}