//
//
//  Perceptron.hpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#ifndef _Perceptron_H_
#define _Perceptron_H_

//#include "Globals.hpp"
//#include "MultiLayerPerceptron.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>           // rand(), srand()
#include <time.h>           // time(0) i.e. the current time that seeds srand()
#include <algorithm>        // needed for all STL algorithms

//using FUNC = double (*)(double) ;

enum class ACTIVATION{ Sigmoid, TanH, ReLu } ;

//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
class Perceptron {
    
    friend class MultiLayerPerceptron ;
    
    // ALL values must be Real values (NO Integers)
    std::vector< double > weights ;     // vector of weights--i.e. parameter vector to be trained
    
    int input_size ;    // number of features in the input vector (excluding bias)
    
    std::vector< double > input ;
    double output ;     // neuron's output after passing via activation function
    
    /* The 'activation_flag' will signal the backpropagation algorighm which activation finction was
    used (by the given neuron) so that the proper derivative can be computed */
    ACTIVATION activation_flag ;

public:
    Perceptron( int input_size, ACTIVATION flag ) ;
    
    double run( std::vector< double > inputs ) ;
    
    void set_weights( std::vector< double > init_vec ) ;
    std::vector<double> & get_weights() ;
    void print_weights() ;
};//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#endif /* Perceptron_hpp */
