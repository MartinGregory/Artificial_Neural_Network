//
//  Activation_Function.cpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 5/1/23.
//

#include <cmath>

class Activation_Function {

public:
    //....................................................................................................
    /* The Weighted Sum (Dot Product) 'z' outputed from the Perceptron must now pass via a non-linear
     activation function. One of such functions is a Sigmoid (Logistic) function: 1/(1+e^-z)
     Sigmoid useful properties:
     - sigmoid function is easily differentiable
     - in practice, the sigmoid is NOT commonly used as an activation function. A tanh function almost
     always performs better
     - takes a real value z and maps it into the range [0 … 1]
     - around 0,1 the graph becomes almost linear which has the tendency of ‘squashing’ the outliers towards
     either 0 or 1
     - very high z(s) result in saturation (extremely close to 1) which causes problems in learning */
    static double sigmoid( double z ){
        return 1.0 / ( 1.0 + exp(-z) ) ;
    }
    //....................................................................................................
    /* Another very popular activation function is ReLu(z) = y = max( z,0 )
     ReLu useful properties:
     - while Tanh and Sigmoid functions map very high values of z into y being close to 1 ; ReLU does NOT
     have this problem
     - ReLU is very close to being linear thus exhibiting properties similar to those of a linear function
     - the most commonly used activation function
     - output is either same as z or 0 ; thus maps z into range [0 … z] */
    static double relu( double z ){
        return fmax( z,0 );
    }
    //....................................................................................................
    /* Tanh useful properties:
     - Tanh function is smoothly differentiable and maps the outliers towards the mean
     - almost always performs better than Sigmoid
     - maps z into range [-1 … +1]
     - very high z(s) result in saturation (extremely close to 1) which causes problems in learning */
    static double tanh(double z) {
        return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
    }
    //....................................................................................................
};
