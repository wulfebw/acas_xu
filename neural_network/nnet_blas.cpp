#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cblas.h>
#include "nnet_blas.hpp"

//Neural Network Struct
class NNet {
public:
    int symmetric;     //1 if network is symmetric, 0 otherwise
    int numLayers;     //Number of layers in the network
    int inputSize;     //Number of inputs to the network
    int outputSize;    //Number of outputs to the network
    int maxLayerSize;  //Maximum size dimension of a layer in the network
    int *layerSizes;   //Array of the dimensions of the layers in the network
    double *means;     //Array of the means used to scale the inputs and outputs
    double *ranges;    //Array of the ranges used to scale the inputs and outputs
    double ***matrix; //4D jagged array that stores the weights and biases
                       //the neural network. 
    double *inputs;    //Scratch array for inputs to the different layers
    double *temp;      //Scratch array for outputs of different layers
};

//Take in a .nnet filename with path and load the network from the file
//Inputs:  filename - const char* that specifies the name and path of file
//Outputs: void *   - points to the loaded neural network
void *load_network(const char* filename)
{
    //Load file and check if it exists
    FILE *fstream = fopen(filename,"r");
    if (fstream == NULL)
    {
        return NULL;
    }

    //Initialize variables
    int bufferSize = 10240;
    char *buffer = new char[bufferSize];
    char *record, *line;
    int i=0, j=0, layer=0, param=0; //int row = 0;
    NNet *nnet = new NNet();

    //Read int parameters of neural network
    line=fgets(buffer,bufferSize,fstream); //skip header line
    line=fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    nnet->numLayers    = atoi(record);
    nnet->inputSize    = atoi(strtok(NULL,",\n"));
    nnet->outputSize   = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = new int[(((nnet->numLayers)+1))];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->numLayers)+1); i++)
    {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }

    //Load the symmetric paramter
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    nnet->symmetric = atoi(record);

    nnet->means = new double[(((nnet->inputSize)+1))];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->inputSize)+1); i++)
    {
        nnet->means[i] = atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->ranges = new double[(((nnet->inputSize)+1))];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->inputSize)+1); i++)
    {
        nnet->ranges[i] = atof(record);
        record = strtok(NULL,",\n");
    }

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the row major matrix
    //
    nnet->matrix = new double **[nnet->numLayers];
    for (layer = 0; layer<(nnet->numLayers); layer++)
    {
        nnet->matrix[layer] = new double*[2];
        nnet->matrix[layer][0] = new double[nnet->layerSizes[layer+1]*nnet->layerSizes[layer]];
        nnet->matrix[layer][1] = new double[nnet->layerSizes[layer+1]];
    }

    //Iteration parameters
    layer = 0;
    param = 0;
    i=0;
    j=0;

    //Read in parameters and put them in the matrix
    while((line=fgets(buffer,bufferSize,fstream))!=NULL)
    {
        if(i>=nnet->layerSizes[layer+1])
        {
            if (param==0)
            {
                param = 1;
            }
            else
            {
                param = 0;
                layer++;
            }
            i=0;
            j=0;
        }
        record = strtok(line,",\n");
        while(record != NULL)
        {
            nnet->matrix[layer][param][j++] = atof(record);
            record = strtok(NULL,",\n");
        }
        i++;
    }
    nnet->inputs = new double[500*nnet->maxLayerSize];
    nnet->temp = new double[500*nnet->maxLayerSize];


    delete[] buffer;
    //return a pointer to the neural network
    return static_cast<void*>(nnet);   
}

//Deallocate memory used by a neural network
//Inputs:  void *network: Points to network struct
//Output:  void
void destroy_network(void *network)
{
    int i=0;//, row=0;
    if (network!=NULL)
    {
        NNet *nnet = static_cast<NNet*>(network);
        for(i=0; i<(nnet->numLayers); i++)
        {
            //free pointer to weights and biases
            delete[] nnet->matrix[i][0];
            delete[] nnet->matrix[i][1];

            //free pointer to the layer of the network
            delete[] nnet->matrix[i];
        }

        //free network parameters and the struct
        delete[] nnet->layerSizes;
        delete[] nnet->means;
        delete[] nnet->ranges;
        delete[] nnet->matrix;
        delete[] nnet->inputs;
        delete[] nnet->temp;
        delete(nnet);
    }
}


//Complete one forward pass for a given set of inputs and return Q values
//Inputs:  void *network - pointer to the neural net struct, obtained from calling "load_network"
//         double *input - double pointer to the inputs to the network
//                             The inputs should be in form [r,th,psi,vOwn,vInt,tau,pa]
//                             with the angles being in radians
//         double *output - double pointer to the outputs from the network
//                              This is where the outputs will be written
//
//Output:  int - 1 if the forward pass was successful, -1 otherwise
int evaluate_network(void *network, double *input, double *output)
{
    int i,layer;
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    //Cast void* to nnets struct pointer
    NNet *nnet = static_cast<NNet*>(network);
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int symmetric    = nnet->symmetric;
    double ***matrix = nnet->matrix;
    
    //Normalize inputs
    for (i=0; i<inputSize;i++)
    {
        nnet->inputs[i] = (input[i]-nnet->means[i])/(nnet->ranges[i]);
    }
    if (symmetric==1 && nnet->inputs[2]<0)
    {
        nnet->inputs[2] = -nnet->inputs[2]; //Make psi positive
        nnet->inputs[1] = -nnet->inputs[1]; //Flip across x-axis
    } else {
        symmetric = 0;
    }

    for (layer = 0; layer<(numLayers); layer++)
    {
        double *weights = matrix[layer][0];
        double *biases  = matrix[layer][1];
        for (i=0; i<nnet->layerSizes[layer+1];i++){
            nnet->temp[i]=biases[i];
        }

        //Matrix - vector multiplication
        cblas_dgemv(CblasRowMajor, CblasNoTrans, nnet->layerSizes[layer+1],nnet->layerSizes[layer],1.0,weights,nnet->layerSizes[layer],nnet->inputs,1,1.0,nnet->temp,1);

        for (i=0; i<nnet->layerSizes[layer+1]; i++)
        {            
            //Perform ReLU
            if (nnet->temp[i]<0 && layer<(numLayers-1))
            {
                nnet->inputs[i] = 0.0;
            } else {
                nnet->inputs[i] = nnet->temp[i];
            }
        }
    }

    //Write the final output value to the allocated spot in memory
    for (i=0; i<outputSize; i++)
    {
        output[i] = nnet->inputs[i]*nnet->ranges[inputSize]+nnet->means[inputSize];
    }

    //If symmetric, switch the Qvalues of actions -1.5 and 1.5 as well as -3 and 3
    if (symmetric == 1){
        double tempValue = output[1];
        output[1] = output[2];
        output[2] = tempValue;
        tempValue = output[3];
        output[3] = output[4];
        output[4] = tempValue;
    }

    //Return 1 for success
    return 1;
}

//Return the number of inputs to a network
//Inputs: void *network - pointer to a network struct
//Output: int - number of inputs to the network, -1 if the network is NULL
int num_inputs(void *network)
{
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    NNet *nnet = static_cast<NNet*>(network);
    return nnet->inputSize;
}

//Return the number of outputs from a network
//Inputs: void *network - pointer to a network struct
//Output: int - number of outputs from the network, -1 if the network is NULL
int num_outputs(void *network)
{
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    NNet *nnet = static_cast<NNet*>(network);
    return nnet->outputSize;
}

//Complete Multiple forward passes for a given set of inputs and returns Q values
//Inputs:  void *network - pointer to the neural net struct, created by calling load_network
//         double *input - double pointer to the inputs to the network. The inputs should
//                             follow one after another, such as:
//                             [r1,th1,psi1,vOwn1,vInt1,tau1,pa1,r2,th2,psi2,vOwn2,vInt2,tau2,pa2,...paN]
//         double *output - double pointer to the outputs from the network, which will be in the form:
//                             [q0_0,q1_0,q2_0,q3_0,q4_0,q0_1,q1_1,q2_1,q3_1,q4_1,...q4_N]
//
//Output:  int - 1 if the forward pass was successful, -1 otherwise
int evaluate_network_multiple(void *network, double *input, int numberInputs, double *output)
{
    int i,j,layer;
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    //Cast void* to nnets struct pointer
    NNet *nnet = static_cast<NNet*>(network);
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int symmetric    = nnet->symmetric;
    int symmetryVec[numberInputs];


    double ***matrix = nnet->matrix;

    //Normalize inputs
    int ind=0;
    for (i=0; i<inputSize;i++) 
    {
        for (j=0; j<numberInputs;j++)
        {
            nnet->inputs[ind] = (input[i+j*inputSize]-nnet->means[i])/(nnet->ranges[i]);
            ind++;
        }
    }


    for (i=0; i<numberInputs; i++){
        if (symmetric==1 && input[i*inputSize+2]<0)
        {
            nnet->inputs[i+numberInputs*2] = (-input[2+i*inputSize]-nnet->means[2])/(nnet->ranges[2]); //Make psi positive
            nnet->inputs[i+numberInputs*1] = (-input[1+i*inputSize]-nnet->means[1])/(nnet->ranges[1]); //Flip across x-axis
            symmetryVec[i] = 1;
        } else {
            symmetryVec[i] = 0;
        }
    }
    
    for (layer = 0; layer<(numLayers); layer++)
    {
        double *weights = matrix[layer][0];
        double *biases  = matrix[layer][1];

        ind = 0;

        //Create matrix to store output of matrix algebra and initialize it to bias values
        for (i=0; i<nnet->layerSizes[layer+1];i++)
        {        
            for(j=0; j<numberInputs; j++)
            {
                nnet->temp[ind]=biases[i];
                ind++;
            }
        }

        //Matrix algebra
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nnet->layerSizes[layer+1],numberInputs,nnet->layerSizes[layer],1.0,weights,nnet->layerSizes[layer],nnet->inputs,numberInputs,1.0,nnet->temp,numberInputs);

        for (i=0; i<nnet->layerSizes[layer+1]*numberInputs; i++)
        {            
            //Perform ReLU
            if (nnet->temp[i]<0 && layer<(numLayers-1))
            {
                nnet->inputs[i] = 0.0;
            } else {
                nnet->inputs[i] = nnet->temp[i];
            }
        }
    }

    //Write the final output value to the allocated spot in memory
    ind = 0;
    for (j=0; j<numberInputs; j++)
    {
        for (i=0; i<outputSize; i++)
        {
            output[ind] = nnet->inputs[i*numberInputs+j]*nnet->ranges[inputSize]+nnet->means[inputSize];
            ind++;
        }
    }

    //Go through the inputs and check if any of them were symmetric
    //If they were, the Q values -1.5 and 1.5, -3 and 3 need to be switched
    double tempValue;
    for (i=0; i<numberInputs; i++){
        if (symmetryVec[i]==1){
            tempValue = output[i*outputSize+1];
            output[i*outputSize+1] = output[i*outputSize+2];
            output[i*outputSize+2] = tempValue;
            tempValue = output[i*outputSize+3];
            output[i*outputSize+3] = output[i*outputSize+4];
            output[i*outputSize+4] = tempValue;
        }
    }
    //Return 1 for success
    return 1;
}