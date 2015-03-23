#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>

using namespace std;



/////// ******** CLASS Neuron ************* ////////
class Neuron;

typedef vector<Neuron> Layer;

class Connection {
public:
    Connection();
    double getWeight(){return weights;}
private:
    double weights;
    double deltaWeights;
    static double randomWeight(void);

};

Connection::Connection() {
    weights = randomWeight();
}

static double Connection::randomWeight(void) {
    return rand() / double(RAND_MAX);
}


class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) {m_outputVal = val; }
    double getOutputVal() const {return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    unsigned getIndex() const {return m_myIndex;}
private:
    double m_outputVal;
    vector<Connection> m_outputWeights;
    int m_myIndex;

    static double transferFunctionDerivative(double sum);
    static double transferFunction(double sum);
};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (int i = 0; i < numOutputs; ++i) {
        m_outputWeights.push_back(Connection());
    }
    m_myIndex = myIndex;
}





/////// ******** CLASS NET ************* ////////

class Net {
public:
    Net(const vector<unsigned> &topology);

    void feedforward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals){};

private:
    vector<Layer> m_layers;
    double m_error;
};

Net::Net(const vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned i = 0; i <  numLayers; ++i) {
        m_layers.push_back(Layer());
        unsigned size = topology[i];
        unsigned sizeOfNextLayer = i == numLayers - 1 ? 0 : topology[i+1];

        //We've created the layer - lets fill with neurons + the bias
        for (unsigned j = 0; j <= size; ++j) {
            m_layers.back().push_back(Neuron(size - 1, j));
            cout << "we've created a neuron" << endl;
        }
    }
};


void Net::feedforward(const vector<double> &inputVals) {
    // if the number of input values are the same as the number of input neurons -> we're good
    assert(inputVals.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }


    // Forward for each neuron
    for (int j = 1; j < m_layers.size(); ++j) {
        Layer &prevLayer = m_layers[j - 1];
        for (int i = 0; i < m_layers[j].size() - 1; ++i) {
            m_layers[j][i].feedForward(prevLayer);
        }

    }
}


int main() {

    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(3);
    topology.push_back(1);
    // e.g {4,2,1} which means 3 layers à 4, 2 ,1 Neurons
    Net myNet(topology);

    vector<double> inputVals;
    myNet.feedforward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultVals;
    myNet.getResults(resultVals);

    cout << "Hello, World!" << endl;
    return 0;
}


void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0;

    // Sum the previous Layers(Neurons) outputs , our inputs
    // Include bias node
    for (int i = 0; i < prevLayer.size(); ++i) {
        sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outputWeights[getIndex()].getWeight();
    }
    m_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double sum) {
    // tanh output range -[1.0 ... 1.0]
    return tanh(sum);
}
double Neuron::transferFunctionDerivative(double sum) {
    return 1 - (sum*sum);
}

/**
* Calculates overall net errors (RMS of output neuron errors)
* Calculates output layer gradient
* Calculates gradients on hidden layers
* For all Layers from output to first hidden layer,
* update all Connection weights
*/
void Net::backProp(const vector<double> &targetVals) {
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (int i = 0; i < outputLayer.size() - 1; ++i) {
        double delta = targetVals[i] - outputLayer[i].getOutputVal();
        m_error += delta*delta;
    }
    m_error /= outputLayer.size() -1;
    m_error = sqrt(m_error); // RMS


}
