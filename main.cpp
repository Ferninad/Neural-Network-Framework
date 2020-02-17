#include "common.h"
#include "cmath"
#include "vector"
#include "string"
#include "random"
#include "OpenSimplexNoise.h"
#include "cstdlib"

bool Init();
void CleanUp();
void Run();
void PrintVector(vector<vector<double>> mat);
vector<vector<vector<double>>> CreateData();
double ScaleNum(double n, double minN, double maxN, double min, double max);

vector<vector<double>> MultMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2);
vector<vector<double>> AddMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2);
vector<vector<double>> HadMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2);
vector<vector<double>> SubMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2);
vector<vector<double>> TransMatrix(vector<vector<double>> matrix);

SDL_Window *window;
SDL_GLContext glContext;
SDL_Surface *gScreenSurface = nullptr;
SDL_Renderer *renderer = nullptr;
SDL_Rect pos;

int screenWidth = 500;
int screenHeight = 500;
double featureSize = 50;
OpenSimplexNoise *noise1 = nullptr;

class NeuronLayer{
    public:
        NeuronLayer(vector<int> layers, int layer, vector<double> biase1); //layer structures, layer neuron is located
        vector<vector<double>> Pass(vector<vector<double>> inputs); //returns outputs, sigmoid(sum of inputs*weights)
        normal_distribution<double> distribution {0.0, 1.0};
        default_random_engine generator {static_cast<unsigned>(1)};
        int layer; //layer
        vector<vector<double>> biase;
        vector<vector<double>> weights; //weights to input data
        vector<vector<double>> output; //output of the layer
    private:
};

class NN{
    public:
        NN(vector<int> layers, vector<vector<double>> biases); //layer structures
        void Pass(vector<vector<double>> inputValues, vector<vector<double>> trainingData); //starting input data, correct ouput data(training data)
        void BackProp(); //changes the weights according to output
        vector<NeuronLayer> neuronLayers; //[layer][neurons in layer]
        vector<int> layers;
        vector<vector<double>> inputValues; //holds starting input data
        vector<vector<double>> trainingData; //holds training data
        vector<vector<double>> output; //holds the passes output
        vector<vector<double>> error; // holds the passes error
        double cost; //cost of the network
        double LearningRate = .1;
    private:
};

void Draw(vector<vector<double>> inputs, NN network);

NN :: NN(vector<int> layers1, vector<vector<double>> biases){ //layer structures and biases
    layers = layers1;
    
    for(int i = 1; i < layers.size(); i++){ //layers, i starts at one because first stuff is inputs
        NeuronLayer layer(layers, i, biases[i-1]);
        neuronLayers.push_back(layer); //stores layer of neurons
    }
}

void NN :: Pass(vector<vector<double>> InputValues, vector<vector<double>> TrainingData){
    inputValues = InputValues;
    trainingData = TrainingData;
    vector<vector<double>> layerInputs;
    vector<vector<double>> layerOutputs;
    layerInputs = inputValues;
    for(int i = 0; i < neuronLayers.size(); i++){
        layerOutputs = neuronLayers[i].Pass(layerInputs);
        layerInputs = layerOutputs;
        layerOutputs.clear();
    }
    output = layerInputs; //output of the pass
}

void NN :: BackProp(){
    error = SubMatrixs(trainingData, output); //calc the errors correct output minus received output
    cost = 0;
    vector<vector<double>> TrainingError = TransMatrix(error);
    vector<vector<double>> nextLayerError;
    for(int i = 0; i < error.size(); i++){ //runs through the specfic neurons. calcs cost of the epoch
        double avgError = 0;
        for(int j = 0; j < error[i].size(); j++){
            avgError += error[i][j]; //sums the average error for each output from the neuron
        }
        cost += pow(avgError/error.size(), 2);
    }
    vector<vector<vector<double>>> adjustWeights; //holds the adjust to the weights for each layer
    vector<vector<vector<double>>> adjustBiases;
    for(int i = 0; i < neuronLayers.size(); i++){ //creates the 2d vectors to hold the weight adjusts and biase adjusts
        adjustWeights.push_back(neuronLayers[i].weights);
        adjustBiases.push_back(neuronLayers[i].biase);
    }
    for(int i = 0; i < adjustWeights.size(); i++){ //sets alls values to 0
        for(int j = 0; j < adjustWeights[i].size(); j++){
            for(int k = 0; k < adjustWeights[i][j].size(); k++){
                adjustWeights[i][j][k] = 0; //sets the adjust of the weights to zero. haven't calced yet
            }
        }
    }
    for(int i = 0; i < adjustBiases.size(); i++){ //sets alls values to 0
        for(int j = 0; j < adjustBiases[i].size(); j++){
            for(int k = 0; k < adjustBiases[i][j].size(); k++){
                adjustBiases[i][j][k] = 0; //sets the adjust of the biase to zero. haven't calced yet
            }
        }
    }
    for(int i = neuronLayers.size()-1; i >= 0; i--){
        for(int j = 0; j < inputValues.size(); j++){ //runs through each training inputs output
            vector<vector<double>> LayerError;
            for(int k = 0; k < TrainingError.size(); k++){
                LayerError.push_back({TrainingError[k][j]});
            }
            if(i > 0){ //only need to calc next layer error if their is a next layer
                nextLayerError = MultMatrixs(TransMatrix(neuronLayers[i].weights), LayerError); //calc next layer error before changing weights
            }
            vector<vector<double>> ChangeWeights; //below calcs the change in weights for the layer
            vector<vector<double>> ChangeBiases;
            vector<vector<double>> gradient;
            gradient.push_back(neuronLayers[i].output[j]);
            gradient = TransMatrix(gradient);
            for(int m = 0; m < gradient.size(); m++){
                for(int n = 0; n < gradient[m].size(); n++){
                    gradient[m][n] = gradient[m][n] * (1 - gradient[m][n]); //derivative of the output
                }
            }
            gradient = HadMatrixs(LayerError, gradient);
            for(int m = 0; m < gradient.size(); m++){
                for(int n = 0; n < gradient[m].size(); n++){
                    gradient[m][n] = gradient[m][n] * LearningRate; //multiply by learning rate
                }
            }
            ChangeBiases = gradient;
            if(i > 0) //checks if the layers input is the trainging input
                ChangeWeights = MultMatrixs(gradient, {neuronLayers[i-1].output[j]});
            else //first layers input is the training input
                ChangeWeights = MultMatrixs(gradient, {inputValues[j]});
            adjustWeights[i] = AddMatrixs(adjustWeights[i], ChangeWeights);
            adjustBiases[i] = AddMatrixs(adjustBiases[i], ChangeBiases);
        }
        TrainingError = nextLayerError;
        nextLayerError.clear();
    }
    for(int i = 0; i < adjustWeights.size(); i++){
        neuronLayers[i].weights = AddMatrixs(neuronLayers[i].weights, adjustWeights[i]);
    }
    for(int i = 0; i < adjustBiases.size(); i++){
        neuronLayers[i].biase = AddMatrixs(neuronLayers[i].biase, adjustBiases[i]);
    }
}

NeuronLayer :: NeuronLayer(vector<int> layers, int layer1, vector<double> biase1){ //layer structures, neuron layer
    layer = layer1;
    biase = {biase1};
    for(int i = 0; i < layers[layer]; i++){
        vector<double> temp;
        for(int j = 0; j < layers[layer-1]; j++){
            temp.push_back(distribution(generator)/(pow(layers[layer-1], .5))); //random number using standard normal deviation
            /*static_cast<double>(rand())/RAND_MAX * 2 - 1*/
        }
        weights.push_back(temp);
    }
}

vector<vector<double>> NeuronLayer :: Pass(vector<vector<double>> inputs){
    vector<vector<double>> result = MultMatrixs(inputs, TransMatrix(weights)); //sum of inputs * weight
    for(int i = 0; i < result.size(); i++){
        for(int j = 0; j < result[i].size(); j++){
            result[i][j] += biase[0][j]; //adds the biase for the neuron
            
            result[i][j] = 1 / (1 + exp(-1*result[i][j])); //result = sigmoid(result)
        }
    }
    output = result;
    return result;
}

bool Init()
{
    if (SDL_Init(SDL_INIT_NOPARACHUTE & SDL_INIT_EVERYTHING) != 0)
    {
        SDL_Log("Unable to initialize SDL: %s\n", SDL_GetError());
        return false;
    }
    else
    {
        //Specify OpenGL Version (4.2)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_Log("SDL Initialised");
    }

    //Create Window Instance
    window = SDL_CreateWindow(
        "Game Engine",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        screenWidth,
        screenHeight,   
        SDL_WINDOW_OPENGL);

    //Check that the window was succesfully created
    if (window == NULL)
    {
        //Print error, if null
        printf("Could not create window: %s\n", SDL_GetError());
        return false;
    }
    else{
        gScreenSurface = SDL_GetWindowSurface(window);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        SDL_Log("Window Successful Generated");
    }
    //Map OpenGL Context to Window
    glContext = SDL_GL_CreateContext(window);

    return true;
}

int main()
{
    //Error Checking/Initialisation
    if (!Init())
    {
        printf("Failed to Initialize");
        return -1;
    }

    // Clear buffer with black background
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    //Swap Render Buffers
    SDL_GL_SwapWindow(window);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    Run();

    CleanUp();
    return 0;
}

void CleanUp()
{
    //Free up resources
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Run()
{
    bool gameLoop = true;
    srand(time(NULL));
    long rand1 = rand() * (RAND_MAX + 1) + rand();
    noise1 = new OpenSimplexNoise{rand1};
    NN network({2,4,1}, {{0,0,0,0},{0}}); //passes number of nodes per layer (first layer is inputs) and biases
    // vector<vector<vector<double>>> data = CreateData();
    // vector<vector<double>> inputs = data[0];
    // vector<vector<double>> training = data[1];
    // for(int i = 0; i < inputs.size(); i++){
    //     pos.x = inputs[i][0]*4;
    //     pos.y = inputs[i][1]*4;
    //     pos.w = 4;
    //     pos.h = 4;
    //     SDL_SetRenderDrawColor(renderer, 255*training[i][0], 0, 255*training[i][0], 255);
    //     SDL_RenderFillRect(renderer, &pos);
    // }
    // SDL_RenderPresent(renderer);
    vector<vector<double>> inputs = {{0,0},{1,0},{0,1},{1,1}};
    vector<vector<double>> training {{0},{1},{1},{0}};
    cout << "Training..." << endl;
    for(int i = 0; i < 70000; i++){
        int num = rand() % inputs.size();
        network.Pass({inputs[num]}, {training[num]});
        network.BackProp();
        if(i % 100 == 0){
            Draw(inputs, network);
            SDL_RenderPresent(renderer);
        }
        cout << "\t" << i << endl;
        // for(int i = 0; i < network.neuronLayers.size(); i++){
        //     cout << "layer " << i << endl;
        //     PrintVector(network.neuronLayers[i].weights);
        // }
    }
    //network.Pass(inputs, training);
    //cout << "cool" << endl;
    //network.BackProp(); //breaks
    //cout << "well" << endl;
    // cout << "output" << endl;
    // PrintVector(network.output);
    Draw(inputs, network);
    while (gameLoop)
    {   
        // pos.x = 0;
        // pos.y = 0;
        // pos.w = screenWidth;
        // pos.h = screenHeight;
        // SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        // SDL_RenderFillRect(renderer, &pos);

        // SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        
        SDL_RenderPresent(renderer);

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                gameLoop = false;
            }
            if (event.type == SDL_KEYDOWN)
            {
                switch (event.key.keysym.sym){
                    case SDLK_ESCAPE:
                        gameLoop = false;
                        break;
                    default:
                        break;
                }
            }

            if (event.type == SDL_KEYUP)
            {
                switch (event.key.keysym.sym){
                    default:
                        break;
                }
            }
        }
    }
}

void Draw(vector<vector<double>> inputs, NN network){
    for(double x = 0; x < 125; x++){
        for(double y = 0; y < 125; y++){
            double num = static_cast<double>(rand())/RAND_MAX;
            if(num <= .5){
                network.Pass({{x/125,y/125}}, {{1}});
                pos.x = x*4;
                pos.y = y*4;
                pos.w = 4;
                pos.h = 4;
                SDL_SetRenderDrawColor(renderer, 255*network.output[0][0], 255*network.output[0][0], 255*network.output[0][0], 255);
                SDL_RenderFillRect(renderer, &pos);
            }
        }
    }
}

vector<vector<vector<double>>> CreateData(){
    vector<vector<vector<double>>> data;
    vector<vector<double>> input;
    vector<vector<double>> output;
    for(double x = 0; x < 125; x++){
        for(double y = 0; y < 125; y++){
            double num = static_cast<double>(rand())/RAND_MAX;
            if(num < .01){
                double type = (*noise1).eval(static_cast<int>(x)/featureSize, static_cast<int>(y)/featureSize);
                type = ScaleNum(type, -1, 1, 0, 1);
                type = round(type);
                input.push_back({x, y});
                if(type == 0)
                    output.push_back({0,1});
                else if(type == 1)
                    output.push_back({1,0});
            }
        }
    }
    data = {input, output};
    return data;
}

double ScaleNum(double n, double minN, double maxN, double min, double max){
    return (((n - minN) / (maxN - minN)) * (max - min)) + min;
}

vector<vector<double>> MultMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2){
    vector<vector<double>> result;
    vector<double> temp;
    double a = 0;
    for(int j = 0; j < mat1.size(); j++){
        for(int k = 0; k < mat2[0].size(); k++){
            for(int i = 0; i < mat1[j].size(); i++){
                a+= mat1[j][i] * mat2[i][k];
            }
            temp.push_back(a);
            a = 0;
        }
        result.push_back(temp);
        temp.clear();
    }
    return result;
}

vector<vector<double>> AddMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2){
    vector<vector<double>> result;
    vector<double> temp;
    double a = 0;
    for(int i = 0; i < mat1.size(); i++){
        for(int j = 0; j < mat1[i].size(); j++){
            a = mat1[i][j] + mat2[i][j];
            temp.push_back(a);
            a = 0;
        }
        result.push_back(temp);
        temp.clear();
    }
    return result;
}

vector<vector<double>> HadMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2){
    vector<vector<double>> result;
    vector<double> temp;
    double a;
    for(int i = 0; i < mat1.size(); i++){
        for(int j = 0; j < mat1[i].size(); j++){
            a = mat1[i][j] * mat2[i][j];
            temp.push_back(a);
        }
        result.push_back(temp);
        temp.clear();
    }
    return result;
}

vector<vector<double>> SubMatrixs(vector<vector<double>> mat1, vector<vector<double>> mat2){
    vector<vector<double>> result;
    vector<double> temp;
    double a;
    for(int i = 0; i < mat1.size(); i++){
        for(int j = 0; j < mat1[i].size(); j++){
            a = mat1[i][j] - mat2[i][j];
            temp.push_back(a);
        }
        result.push_back(temp);
        temp.clear();
    }
    return result;
}

vector<vector<double>> TransMatrix(vector<vector<double>> matrix){
    vector<vector<double>> result;
    for(int i = 0; i < matrix[0].size(); i++){
        result.push_back({matrix[0][i]});
    }
    for(int i = 1; i < matrix.size(); i++){
        for(int j = 0; j < matrix[i].size(); j++){
            result[j].push_back(matrix[i][j]);
        }
    }
    return result;
}

void PrintVector(vector<vector<double>> mat){
    for(int i = 0; i < mat.size(); i++){
        for(int j = 0; j < mat[i].size(); j++){
            cout << mat[i][j] << "\t";
        }
        cout << endl;
    }
}